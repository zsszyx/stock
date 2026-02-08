import pandas as pd
import sqlite3
from datetime import datetime
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from data_context.context import Minutes5Context, DailyContext
from selector.score_selector import KSPScoreSelector
from sql_op.op import SqlOp
from sql_op import sql_config

def run_ranking():
    print("Initializing Scoring Ranking...")
    sql_op = SqlOp()
    db_file_path = sql_config.db_path.replace("sqlite:// বিশুদ্ধ", "")
    
    # 1. Get latest date
    # Use sql_op.query for consistency
    res = sql_op.query(f"SELECT MAX(date) as max_date FROM {sql_config.mintues5_table_name}")
    if res is None or res.empty or not res.iloc[0]['max_date']:
        print("No data found in database.")
        return

    latest_date_str = res.iloc[0]['max_date']
    latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
    print(f"Latest trading date: {latest_date_str}")

    # 2. Fetch last 5 trading dates
    query = f"""
    SELECT * FROM {sql_config.mintues5_table_name} 
    WHERE date IN (
        SELECT DISTINCT date FROM {sql_config.mintues5_table_name} 
        ORDER BY date DESC LIMIT 5
    )
    """
    print("Fetching data from SQL (Last 5 days)...")
    df = sql_op.query(query)
    
    if df is None or df.empty:
        print("Failed to fetch data.")
        return
    
    print(f"Fetched {len(df)} rows.")

    # 3. Initialize Contexts
    print("Aggregating to Daily Context...")
    min5_ctx = Minutes5Context(df)
    daily_ctx = DailyContext(min5_ctx)
    
    # 4. Run Selector
    selector = KSPScoreSelector(daily_ctx)
    print(f"Calculating scores for {latest_date_str}...")
    
    scores_df = selector.get_scores(latest_date)
    
    if scores_df.empty:
        print("No scores calculated (possibly missing data).")
        return

    # 5. Display Results
    print("\n--- Top 20 High Scores (Positive) ---")
    print("Criteria: Kurt >= 0, Skew <= 0, |PctChg| <= 5%")
    print(scores_df.head(20).to_string(index=False))
    
    print("\n--- Bottom 20 Low Scores (Negative) ---")
    print("Criteria: Kurt < 0 OR Skew > 0 OR |PctChg| > 5%")
    print(scores_df.tail(20).to_string(index=False))

    # Optional: Save to CSV
    # scores_df.to_csv("ksp_scores.csv", index=False)

if __name__ == "__main__":
    run_ranking()
