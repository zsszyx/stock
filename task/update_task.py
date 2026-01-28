import sys
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bao.bao_interface import BaoInterface
from sql_op.op import SqlOp
from sql_op import sql_config

class UpdateTask:
    def __init__(self):
        self.sql_op = SqlOp()

    def fetch_stock_data(self, code, start_date, end_date):
        """
        Worker function to fetch data for a single stock.
        """
        try:
            with BaoInterface() as bi:
                return bi.get_k_data_5min(code=code, start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"Error fetching {code}: {e}")
            return None

    def run_init_mintues5_task(self, start_date: str, end_date: str, max_workers: int = 10, batch_size: int = 50):
        """
        Initializes/Updates 5-minute K-line data with concurrency.
        
        Args:
            date: The reference date (usually today/last trade date) for checking stock list.
            start_date: Default start date if no data exists.
            end_date: End date for data fetching.
            max_workers: Number of parallel fetch threads.
            batch_size: Number of stocks to process and save in one transaction.
        """
        print(f"Starting update task with {max_workers} workers, batch size {batch_size}...")
        
        with BaoInterface() as bi:
            trade_dates = bi.get_trade_dates(start_date=start_date, end_date=end_date)
            if trade_dates.empty:
                print("No trade dates found.")
                return
            last_day = trade_dates.iloc[-1]['calendar_date']
            
            # Get stock list
            stock_list = bi.get_stock_list(date=last_day)
            if stock_list is None:
                raise ValueError("Failed to get stock list")

            # Filter indices
            stock_list = stock_list[~stock_list['code'].str.contains('sh.000|sz.399')]
            all_codes = stock_list['code'].tolist()
            print(f"Total stocks to process: {len(all_codes)}")
        
        with tqdm(total=len(all_codes), desc="Updating Stocks") as pbar:
            for i in range(0, len(all_codes), batch_size):
                batch_codes = all_codes[i:i+batch_size]
                
                # 1. Get existing max dates for this batch
                max_dates = self.sql_op.get_max_date_for_codes(batch_codes, sql_config.mintues5_table_name)
                
                tasks = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for code in batch_codes:
                        current_start = start_date
                        
                        # Incremental update logic
                        if code in max_dates and max_dates[code] is not None:
                            latest_date_in_db = pd.to_datetime(max_dates[code])
                            # If DB is already up to date (or ahead?), skip or just fetch latest day to be safe
                            if latest_date_in_db.strftime('%Y-%m-%d') >= last_day:
                                pbar.update(1)
                                continue
                            current_start = latest_date_in_db.strftime('%Y-%m-%d')
                        
                        # Submit task
                        tasks.append(executor.submit(self.fetch_stock_data, code, current_start, end_date))

                    # Collect results
                    k_data_list = []
                    for future in as_completed(tasks):
                        res = future.result()
                        if res is not None and not res.empty:
                            k_data_list.append(res)
                        pbar.update(1)

                # 2. Save batch to DB
                if k_data_list:
                    all_k_data = pd.concat(k_data_list, ignore_index=True)
                    # Normalize columns if needed? BaoInterface usually returns standard cols
                    all_k_data.set_index(['code', 'time'], inplace=True)
                    self.sql_op.upsert_df_to_db(all_k_data, sql_config.mintues5_table_name, index=True)

if __name__ == '__main__':
    task = UpdateTask()
    # Example usage
    task.run_init_mintues5_task(start_date='2025-12-01', end_date='2026-01-22')