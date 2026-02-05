import pandas as pd
import sqlite3
from datetime import datetime
from data_context.context import Minutes5Context, DailyContext
from selector import (
    MaxDailyPctChgSelector, 
    POCNearSelector, 
    NegativeSkewSelector, 
    TopKurtosisSelector,
    AfternoonStrongSelector,
    VReversalSelector,
    PrevDayNegativeReturnSelector,
    PrevDayAmplitudeSelector
)
from sql_op.op import SqlOp
from sql_op import sql_config

def run_pipeline():
    # 1. Setup Database Connection and fetch data
    sql_op = SqlOp()
    db_file_path = sql_config.db_path.replace("sqlite:///", "")
    
    with sqlite3.connect(db_file_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT MAX(date) FROM {sql_config.mintues5_table_name}")
        latest_date_str = cursor.fetchone()[0]
    
    if not latest_date_str:
        print("No data found in database.")
        return

    latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
    print(f"Latest trading date: {latest_date_str}")

    # Fetch last 15 trading dates
    query = f"""
    SELECT * FROM {sql_config.mintues5_table_name} 
    WHERE date IN (
        SELECT DISTINCT date FROM {sql_config.mintues5_table_name} 
        ORDER BY date DESC LIMIT 15
    )
    """
    print("Fetching data from SQL...")
    with sqlite3.connect(db_file_path) as conn:
        df = pd.read_sql_query(query, conn)
    
    print(f"Fetched {len(df)} rows of 5-minute data.")

    # 2. Initialize Contexts
    print("Initializing Contexts (performing daily aggregation)...")
    min5_ctx = Minutes5Context(df)
    daily_ctx = DailyContext(min5_ctx)
    
    # 3. Selection Pipeline
    print("\n--- Starting Selection Pipeline ---")
    
    # A. Previous Day Amplitude Selection (<= 3%)
    amp_selector = PrevDayAmplitudeSelector(daily_ctx)
    candidate_codes = amp_selector.select(latest_date, threshold=0.03)
    print(f"1. Previous Day Amplitude <= 3%: {len(candidate_codes)} stocks found.")

    if not candidate_codes:
        print("No stocks passed the Amplitude filter.")
        return

    # B. Max Daily Pct Change in recent 5 Days <= 5%
    # This ensures no single day in the last week had a move > 5% or < -5%
    vol_selector = MaxDailyPctChgSelector(daily_ctx)
    candidate_codes = vol_selector.select(latest_date, candidate_codes=candidate_codes, days=5, threshold=0.05)
    print(f"2. Max Daily PctChg in 5D <= 5%: {len(candidate_codes)} stocks remaining.")

    if not candidate_codes:
        print("No stocks passed the MaxDailyPctChg filter.")
        return

    # C. V-Reversal Selection
    v_selector = VReversalSelector(daily_ctx)
    candidate_codes = v_selector.select(latest_date, candidate_codes=candidate_codes, threshold_drop=0.015, threshold_recover=0.015)
    print(f"3. V-Reversal Selected: {len(candidate_codes)} stocks remaining.")

    if not candidate_codes:
        print("No stocks passed the V-Reversal filter.")
        return

    # D. Negative Skew Selection (skew < 0)
    skew_selector = NegativeSkewSelector(daily_ctx)
    candidate_codes = skew_selector.select(latest_date, candidate_codes=candidate_codes)
    print(f"4. Negative Skew (skew < 0): {len(candidate_codes)} stocks remaining.")

    if not candidate_codes:
        print("No stocks passed the Negative Skew filter.")
        return

    # E. POC Near Selection (close > poc * 0.99)
    poc_selector = POCNearSelector(daily_ctx)
    candidate_codes = poc_selector.select(latest_date, candidate_codes=candidate_codes, threshold=0.01)
    print(f"5. Close > POC * 0.99: {len(candidate_codes)} stocks remaining.")

    if not candidate_codes:
        print("No stocks passed the POC filter.")
        return

    # F. Top 10 Kurtosis Selection
    kurt_selector = TopKurtosisSelector(daily_ctx)
    final_codes = kurt_selector.select(latest_date, candidate_codes=candidate_codes, top_n=10)
    print(f"6. Top 10 Kurtosis: Final {len(final_codes)} stocks selected.")

    # 4. Output Results
    if final_codes:
        res_df = daily_ctx.data[
            (daily_ctx.data['code'].isin(final_codes)) & 
            (daily_ctx.data['date'] == latest_date_str)
        ].copy()
        
        # Calculate details for display
        res_df['drop_%'] = (res_df['open'] - res_df['min']) / res_df['open'] * 100
        res_df['recover_%'] = (res_df['close'] - res_df['min']) / res_df['min'] * 100
        
        # Fetch previous day data for amplitude display
        prev_date_str = sorted(daily_ctx.data['date'].unique())[-2]
        prev_df = daily_ctx.data[
            (daily_ctx.data['code'].isin(final_codes)) & 
            (daily_ctx.data['date'] == prev_date_str)
        ][['code', 'amplitude']]
        prev_df.columns = ['code', 'prev_amp']
        
        res_df = res_df.merge(prev_df, on='code')
        
        print("\nSelected Stocks Details:")
        print(res_df[['code', 'open', 'min', 'close', 'min_time', 'skew', 'prev_amp', 'drop_%', 'recover_%']].head(20))
    else:
        print("\nNo stocks matched all criteria.")

if __name__ == "__main__":
    run_pipeline()