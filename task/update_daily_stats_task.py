import sys
import os
import pandas as pd
from sqlalchemy import text

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sql_op.op import SqlOp
from sql_op import sql_config
from strategy.daily_stats_strategy import calculate_and_save_daily_stats

class UpdateDailyStatsTask:
    def __init__(self):
        self.sql_op = SqlOp()

    def run(self):
        print("Checking for daily stats updates...")
        
        # 1. Get max date from daily_stats
        query_max_stats = f"SELECT MAX(date) FROM {sql_config.daily_stats_table_name}"
        try:
            # Check if table exists first (SqlOp creates engine but doesn't guarantee table existence)
            # The simplest way with SqlOp is just to try the query. 
            # If table doesn't exist, it might throw an error.
            res = self.sql_op.query(query_max_stats)
            if res is not None and not res.empty and res.iloc[0, 0] is not None:
                last_stats_date = res.iloc[0, 0]
                print(f"Last stats date: {last_stats_date}")
            else:
                last_stats_date = None
                print("No existing daily stats found.")
        except Exception as e:
            # Table likely doesn't exist
            print(f"Table {sql_config.daily_stats_table_name} check failed (probably doesn't exist): {e}")
            last_stats_date = None

        # 2. Find missing dates in mintues5
        if last_stats_date:
            # Use >= to re-process the last date in case it was partial or updated
            date_query = f"SELECT DISTINCT date FROM {sql_config.mintues5_table_name} WHERE date >= '{last_stats_date}' ORDER BY date"
        else:
            date_query = f"SELECT DISTINCT date FROM {sql_config.mintues5_table_name} ORDER BY date"
            
        dates_df = self.sql_op.query(date_query)
        
        if dates_df is None or dates_df.empty:
            print("No new dates to process.")
            return

        missing_dates = dates_df['date'].astype(str).tolist()
        print(f"Found {len(missing_dates)} missing dates: {missing_dates}")

        # 3. Process day by day to avoid memory explosion
        for date in missing_dates:
            print(f"Processing {date}...")
            calculate_and_save_daily_stats(date, date)
            
        print("Update complete.")

if __name__ == '__main__':
    task = UpdateDailyStatsTask()
    task.run()
