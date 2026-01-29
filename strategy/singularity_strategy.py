import sys
import os
import pandas as pd
import numpy as np

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sql_op.op import SqlOp
from sql_op import sql_config
from strategy.filter import apply_volume_zero_filter

def calculate_poc(df: pd.DataFrame, price_bins: int = 50) -> float:
    """
    Calculates the Point of Control (POC) for the given dataframe.
    POC is the price level with the highest traded volume.
    
    Args:
        df: DataFrame containing 'amount' and 'volume' columns.
        price_bins: Number of bins for the histogram.
        
    Returns:
        float: The POC price.
    """
    # Use average price of the bar for volume accumulation
    if 'avg_price' not in df.columns:
        df = df.copy()
        df['avg_price'] = df['amount'] / df['volume']
        
    hist, bin_edges = np.histogram(
        df['avg_price'], 
        bins=price_bins, 
        weights=df['volume'],
        range=(df['avg_price'].min(), df['avg_price'].max())
    )
    
    if len(hist) == 0:
        return 0.0
        
    max_vol_idx = np.argmax(hist)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers[max_vol_idx]

def run_singularity_strategy(target_date: str = None):
    """
    Executes the 'Singularity' strategy to identify stocks ready for a bullish breakout.
    
    Criteria:
    1. Skewness (S) < -1.0 (P-shape, accumulation)
    2. Kurtosis (K) > 1.0 (Energy compression) and Rising (higher than previous day)
    3. K-line: Low volatility, close slightly above POC.
    """
    sql_operator = SqlOp()
    
    if target_date is None:
        # Default to the latest available date in daily_stats
        date_query = f"SELECT MAX(date) FROM {sql_config.daily_stats_table_name}"
        res = sql_operator.query(date_query)
        if res is not None and not res.empty:
            target_date = pd.to_datetime(res.iloc[0, 0]).strftime('%Y-%m-%d')
        else:
            print("No daily stats found.")
            return

    print(f"Running Singularity Strategy for {target_date}...")
    
    # 1. Fetch potential candidates from daily_stats
    # Criteria: S < -1.0 and K > 1.0
    stats_query = f"""
        SELECT code, weighted_skew, weighted_kurt
        FROM {sql_config.daily_stats_table_name}
        WHERE strftime('%Y-%m-%d', date) = '{target_date}'
          AND weighted_skew < -1.0
          AND weighted_kurt > 1.0
    """
    candidates_df = sql_operator.query(stats_query)
    
    if candidates_df is None or candidates_df.empty:
        print("No candidates found meeting Skewness/Kurtosis criteria.")
        return
        
    print(f"Initial candidates (S < -1.0, K > 1.0): {len(candidates_df)}")
    
    # 2. Check for Rising Kurtosis
    # We need the previous trading day's stats for these candidates
    prev_date_query = f"""
        SELECT MAX(date) FROM {sql_config.daily_stats_table_name} 
        WHERE strftime('%Y-%m-%d', date) < '{target_date}'
    """
    res_prev = sql_operator.query(prev_date_query)
    if res_prev is not None and not res_prev.empty and res_prev.iloc[0, 0] is not None:
        prev_date = pd.to_datetime(res_prev.iloc[0, 0]).strftime('%Y-%m-%d')
        
        prev_stats_query = f"""
            SELECT code, weighted_kurt as prev_kurt
            FROM {sql_config.daily_stats_table_name}
            WHERE strftime('%Y-%m-%d', date) = '{prev_date}'
              AND code IN ({','.join(["'" + c + "'" for c in candidates_df['code']])})
        """
        prev_stats_df = sql_operator.query(prev_stats_query)
        
        if prev_stats_df is not None and not prev_stats_df.empty:
            candidates_df = pd.merge(candidates_df, prev_stats_df, on='code', how='inner')
            # Filter: Rising Kurtosis
            candidates_df = candidates_df[candidates_df['weighted_kurt'] > candidates_df['prev_kurt']]
            print(f"Candidates after 'Rising Kurtosis' check: {len(candidates_df)}")
    else:
        print("Could not find previous date for comparison. Skipping 'Rising Kurtosis' check.")

    final_results = []
    
    # 3. Analyze K-line & POC for remaining candidates
    for _, row in candidates_df.iterrows():
        code = row['code']
        
        # Fetch 5-min data for the target date
        k_data = sql_operator.read_k_data_by_date_range(sql_config.mintues5_table_name, target_date, target_date)
        
        # Filter for specific code
        # DB usually stores code as 'sh.600000', ensure matching
        if k_data.empty:
            continue
            
        # Filter in memory (similar to other strategies)
        if '.' in k_data['code'].iloc[0]:
             k_data_code = k_data[k_data['code'].str.endswith(code)]
        else:
             k_data_code = k_data[k_data['code'] == code]
             
        if k_data_code.empty:
            continue
            
        # Clean data
        k_data_code = apply_volume_zero_filter(k_data_code)
        k_data_code = k_data_code[k_data_code['volume_filter']].copy()
        
        if k_data_code.empty:
            continue
            
        cols_to_numeric = ['open', 'high', 'low', 'close', 'amount', 'volume']
        for col in cols_to_numeric:
            k_data_code[col] = pd.to_numeric(k_data_code[col], errors='coerce')
            
        # Calculate POC
        poc = calculate_poc(k_data_code)
        if poc == 0:
            continue
            
        # Get Daily Close
        daily_close = k_data_code['close'].iloc[-1]
        
        # Check: Close slightly above POC (e.g., 0% to 2% above)
        # "收盘价稳稳站住 POC" -> Close >= POC
        # "收盘价略高于 POC" -> Close <= POC * 1.02 (approx)
        if daily_close < poc:
            continue
            
        if daily_close > poc * 1.02: # Too far gone? "刚启动" usually implies close proximity
            continue
            
        # Check: Low Volatility (Compression)
        # Daily Amplitude = (High - Low) / Close
        daily_high = k_data_code['high'].max()
        daily_low = k_data_code['low'].min()
        amplitude = (daily_high - daily_low) / daily_close
        
        # Threshold: e.g., < 3% amplitude for "Low Volatility"
        if amplitude > 0.03: 
            continue
            
        final_results.append({
            'code': code,
            'skew': row['weighted_skew'],
            'kurt': row['weighted_kurt'],
            'prev_kurt': row.get('prev_kurt', np.nan),
            'close': daily_close,
            'poc': poc,
            'amplitude': amplitude
        })

    # Output Results
    print("\n--- Singularity Strategy Results ---")
    if not final_results:
        print("No stocks met all criteria.")
    else:
        results_df = pd.DataFrame(final_results)
        # Sort by Kurtosis (Energy)
        results_df.sort_values(by='kurt', ascending=False, inplace=True)
        
        print(f"{'Code':<10} {'Skew':<10} {'Kurt':<10} {'Prev K':<10} {'Close':<10} {'POC':<10} {'Amp%':<10}")
        print("-" * 75)
        for _, row in results_df.iterrows():
            print(f"{row['code']:<10} {row['skew']:<10.2f} {row['kurt']:<10.2f} {row['prev_kurt']:<10.2f} {row['close']:<10.2f} {row['poc']:<10.2f} {row['amplitude']*100:<10.2f}")

if __name__ == '__main__':
    run_singularity_strategy()
