import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from data_context.context import Minutes5Context, DailyContext
from selector.score_selector import KSPScoreSelector
from sql_op.op import SqlOp
from sql_op import sql_config

def check_scores():
    target_date_str = "2025-02-06"
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    
    # We need about 2 weeks of data to calculate Kurt/Skew if the context requires it.
    # Looking at DailyContext, it usually aggregates from min5.
    # We'll fetch data around that period.
    
    sql_op = SqlOp()
    
    # Fetch 15 days of data before and including target_date to ensure we have enough for context aggregation
    # Note: KSPScoreSelector uses daily_context.get_window(date, window_days=1)
    # But DailyContext itself needs data to be initialized.
    
    start_date = target_date - timedelta(days=20)
    start_date_str = start_date.strftime("%Y-%m-%d")
    
    print(f"Fetching data from {start_date_str} to {target_date_str}...")
    query = f"""
    SELECT * FROM {sql_config.mintues5_table_name} 
    WHERE date >= '{start_date_str}' AND date <= '{target_date_str}'
    """
    df = sql_op.query(query)
    
    if df is None or df.empty:
        print("No data found for the specified period.")
        return
    
    print(f"Fetched {len(df)} rows.")

    min5_ctx = Minutes5Context(df)
    daily_ctx = DailyContext(min5_ctx)
    
    selector = KSPScoreSelector(daily_ctx)
    print(f"Calculating scores for {target_date_str}...")
    
    scores_df = selector.get_scores(target_date)
    
    if scores_df.empty:
        print(f"No scores calculated for {target_date_str}.")
        return

    print(f"\n--- Top 5 High Scores for {target_date_str} ---")
    print(scores_df.head(5).to_string(index=False))

if __name__ == "__main__":
    check_scores()