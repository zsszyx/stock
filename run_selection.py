import pandas as pd
import sqlite3
from datetime import datetime
from data_context.context import Minutes5Context, DailyContext
from selector import (
    MaxDailyPctChgSelector, 
    POCNearSelector, 
    NegativeSkewSelector, 
    TopKurtosisSelector
)
from sql_op.op import SqlOp
from sql_op import sql_config

def run_pipeline():
    # 1. Setup Database Connection and fetch data
    sql_op = SqlOp()
    
    # Get the latest date from the database
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
    
    # 3. Chain Selection Logic
    print("\n--- Starting Selection Pipeline ---")
    
    # A. Max Daily Pct Change in recent 5 Days <= 5%
    selector_vol = MaxDailyPctChgSelector(daily_ctx)
    candidate_codes = selector_vol.select(latest_date, days=5, threshold=0.05)
    print(f"1. Max Daily PctChg in 5D <= 5%: {len(candidate_codes)} stocks found.")
    
    if not candidate_codes:
        print("No stocks passed the first filter.")
        return

    # B. Close > POC * 0.99
    selector_poc = POCNearSelector(daily_ctx)
    candidate_codes = selector_poc.select(latest_date, candidate_codes=candidate_codes, threshold=0.01)
    print(f"2. Close > POC * 0.99: {len(candidate_codes)} stocks remaining.")

    if not candidate_codes:
        print("No stocks passed the second filter.")
        return

    # C. Negative Skew (skew < 0)
    selector_skew = NegativeSkewSelector(daily_ctx)
    candidate_codes = selector_skew.select(latest_date, candidate_codes=candidate_codes)
    print(f"3. Negative Skew: {len(candidate_codes)} stocks remaining.")

    if not candidate_codes:
        print("No stocks passed the third filter.")
        return

    # D. Top 5 Kurtosis
    selector_kurt = TopKurtosisSelector(daily_ctx)
    final_codes = selector_kurt.select(latest_date, candidate_codes=candidate_codes, top_n=10)
    print(f"4. Top 5 Kurtosis: Final {len(final_codes)} stocks selected.")

    # 4. Output Results
    if final_codes:
        print("\nSelected Stocks:")
        final_df = daily_ctx.data[
            (daily_ctx.data['code'].isin(final_codes)) & 
            (daily_ctx.data['date'] == latest_date_str)
        ].sort_values('kurt', ascending=False)
        
        print(final_df[['code', 'date', 'close', 'poc', 'skew', 'kurt', 'pct_chg']])
    else:
        print("\nNo stocks matched all criteria.")

if __name__ == "__main__":
    run_pipeline()