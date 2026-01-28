import pandas as pd
import numpy as np
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sql_op import op, sql_config
from strategy.filter import apply_volume_zero_filter

def calculate_weighted_stats(values, weights):
    """
    Computes weighted skewness and weighted kurtosis (Fisher, excess).
    """
    if len(values) < 2 or weights.sum() == 0:
        return 0.0, 0.0

    # Weighted Mean
    mean = np.average(values, weights=weights)
    
    # Weighted Variance
    variance = np.average((values - mean)**2, weights=weights)
    
    if variance == 0:
        return 0.0, 0.0

    std_dev = np.sqrt(variance)
    
    # Weighted Skewness
    # Formula: sum(w * (x - mean)^3) / (sum(w) * std^3)
    skew = np.average(((values - mean) / std_dev)**3, weights=weights)
    
    # Weighted Kurtosis (Excess)
    # Formula: sum(w * (x - mean)^4) / (sum(w) * std^4) - 3
    kurt = np.average(((values - mean) / std_dev)**4, weights=weights) - 3
    
    return skew, kurt

import pandas as pd
import numpy as np
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sql_op import op, sql_config
from strategy.filter import apply_volume_zero_filter

def calculate_weighted_stats(values, weights):
    """
    Computes weighted skewness and weighted kurtosis (Fisher, excess).
    """
    if len(values) < 2 or weights.sum() == 0:
        return 0.0, 0.0

    # Weighted Mean
    mean = np.average(values, weights=weights)
    
    # Weighted Variance
    variance = np.average((values - mean)**2, weights=weights)
    
    if variance == 0:
        return 0.0, 0.0

    std_dev = np.sqrt(variance)
    
    # Weighted Skewness
    # Formula: sum(w * (x - mean)^3) / (sum(w) * std^3)
    skew = np.average(((values - mean) / std_dev)**3, weights=weights)
    
    # Weighted Kurtosis (Excess)
    # Formula: sum(w * (x - mean)^4) / (sum(w) * std^4) - 3
    kurt = np.average(((values - mean) / std_dev)**4, weights=weights) - 3
    
    return skew, kurt

def calculate_and_save_daily_stats(start_date: str, end_date: str):
    """
    Calculates daily stats for the given date range and saves to DB.
    """
    sql_operator = op.SqlOp()
    print(f"Calculating stats from {start_date} to {end_date}...")
    
    # 2. Read data
    k_data = sql_operator.read_k_data_by_date_range(sql_config.mintues5_table_name, start_date, end_date)
    
    if k_data.empty:
        print("No K-line data found.")
        return

    # Normalize data
    # Remove 'sh.'/'sz.' prefix from code
    k_data['code'] = k_data['code'].astype(str).apply(lambda x: x.split('.')[-1])
    
    # Convert numeric columns
    cols_to_numeric = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in cols_to_numeric:
        k_data[col] = pd.to_numeric(k_data[col], errors='coerce')

    # Filter invalid data
    k_data = apply_volume_zero_filter(k_data)
    k_data = k_data[k_data['volume_filter']]
    
    # Calculate 'price' for the bar (VWAP of the bar)
    k_data['bar_price'] = k_data['amount'] / k_data['volume']

    results = []

    # 3. Group by Date and Code
    # This might take a moment if data is large
    print(f"Processing {len(k_data)} records...")
    grouped = k_data.groupby(['date', 'code'])

    for (date, code), group in grouped:
        if len(group) < 3: # Need at least a few bars to calculate stats
            continue
            
        prices = group['bar_price']
        volumes = group['volume']
        
        # --- Standard Stats (Unweighted) ---
        # Pandas skew/kurt are unbiased estimators by default
        std_skew = prices.skew()
        std_kurt = prices.kurt()
        
        # --- Weighted Stats ---
        w_skew, w_kurt = calculate_weighted_stats(prices.values, volumes.values)
        
        results.append({
            'date': date,
            'code': code,
            'skew': std_skew,
            'kurt': std_kurt,
            'weighted_skew': w_skew,
            'weighted_kurt': w_kurt,
            'volume_sum': volumes.sum()
        })

    if not results:
        print("No stats calculated.")
        return

    # 4. Create DataFrame
    stats_df = pd.DataFrame(results)
    
    # Handle NaNs (e.g., if std dev was 0)
    stats_df = stats_df.fillna(0)
    
    # Save to DB
    print(f"Saving {len(stats_df)} records to {sql_config.daily_stats_table_name}...")
    # Setting index to False because we are not using the DataFrame index as a DB column, 
    # but we might want a composite primary key (date, code) later. 
    # For now, let's just dump it. upsert_df_to_db usually handles index=False fine if we don't rely on index.
    # However, to support upsert correctly, we usually need a unique constraint or primary key.
    # The current `upsert_df_to_db` implementation in `op.py` is a bit generic (delete+insert or replace).
    # Let's check `op.py` again. It uses a temp table and INSERT OR REPLACE.
    # It requires the target table to have a PRIMARY KEY or UNIQUE constraint for REPLACE to work effectively as an update.
    # If the table is created by `to_sql`, it might not have PKs.
    # But let's proceed. 
    sql_operator.upsert_df_to_db(stats_df, sql_config.daily_stats_table_name, index=False)
    print("Done.")

def run_daily_stats_strategy():
    """
    Manual run: Calculates daily Skewness and Kurtosis for the last 10 days.
    """
    sql_operator = op.SqlOp()
    
    # 1. Get the last 10 days of data (adjustable)
    print("Fetching available dates...")
    date_query = f"SELECT DISTINCT date FROM {sql_config.mintues5_table_name} ORDER BY date DESC LIMIT 10"
    dates_df = sql_operator.query(date_query)
    
    if dates_df is None or dates_df.empty:
        print("No data found in database.")
        return

    dates = sorted(dates_df['date'].astype(str).tolist())
    start_date = dates[0]
    end_date = dates[-1]
    
    calculate_and_save_daily_stats(start_date, end_date)

if __name__ == '__main__':
    run_daily_stats_strategy()
