import sys
import os

# Add the project root to sys.path
# File is at: .../stock/stock/stock/data_fetch/task/update_min5_task.py
# The 'stock' package is at: .../stock/stock/stock/
# We need to add the parent: .../stock/stock/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from data_fetch.data_provider.baostock_provider import BaoInterface
from sql_op.op import SqlOp
from sql_op import sql_config

# Global instance for worker processes
worker_bi = None

def init_worker():
    """
    Initialize the BaoInterface in the worker process.
    This creates a persistent session for the lifetime of the worker.
    """
    global worker_bi
    worker_bi = BaoInterface()
    # Manually login. Accessing protected method _login is necessary here 
    # to avoid the context manager's auto-logout.
    worker_bi._login()

def fetch_data_task(code, start_date, end_date, adjustflag='3'):
    """
    Module-level task function for multiprocessing.
    Uses the global worker_bi instance.
    """
    global worker_bi
    try:
        if worker_bi is None:
            # Fallback if init didn't run (shouldn't happen with correct Executor setup)
            init_worker()
        return worker_bi.get_k_data_5min(code=code, start_date=start_date, end_date=end_date, adjustflag=adjustflag)
    except Exception as e:
        # In multiprocessing, print might not show up immediately or at all depending on config
        # but it's better than silent failure.
        return None

class UpdateTask:
    def __init__(self):
        self.sql_op = SqlOp()

    def run_init_mintues5_task(self, start_date: str, end_date: str, max_workers: int = 6, batch_size: int = 50, adjustflag: str = "3"):
        """
        Initializes/Updates 5-minute K-line data using Multiprocessing.
        
        Args:
            start_date: Default start date if no data exists.
            end_date: End date for data fetching.
            max_workers: Number of parallel processes.
            batch_size: Number of stocks to process and save in one transaction.
            adjustflag: 3 for unadjusted, 2 for forward adjusted.
        """
        # Note: Reduced default max_workers to 6 to be conservative with system resources
        print(f"Starting update task with {max_workers} processes, batch size {batch_size}, adjustflag {adjustflag}...")
        
        # We need a temporary BaoInterface in the main process just to get the stock list
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

            # Filter: Only Main Board (sh.60, sz.00) and ChiNext (sz.30)
            stock_list = stock_list[stock_list['code'].str.match(r'^(sh\.60|sz\.00|sz\.30)')]
            
            # --- NEW: Filter out 'ST', 'S', 'T', '*ST' stocks ---
            original_count = len(stock_list)
            st_t_s_filter = (
                (stock_list['code_name'].str.contains('ST', case=False, na=False)) |
                (stock_list['code_name'].str.contains('\\*ST', case=False, na=False)) |
                (stock_list['code_name'].str.contains('S', case=False, na=False)) |
                (stock_list['code_name'].str.contains('T', case=False, na=False))
            )
            stock_list = stock_list[~st_t_s_filter] # Keep stocks NOT matching the filter
            print(f"Filtered out {original_count - len(stock_list)} 'ST/S/T/*ST' stocks.")
            
            all_codes = stock_list['code'].tolist()
            print(f"Total stocks to process: {len(all_codes)}")
        
        # Use ProcessPoolExecutor for BaoStock interactions
        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
            with tqdm(total=len(all_codes), desc="Updating Stocks") as pbar:
                for i in range(0, len(all_codes), batch_size):
                    batch_codes = all_codes[i:i+batch_size]
                    
                    # Get existing date ranges for this batch
                    max_dates = self.sql_op.get_max_date_for_codes(batch_codes, sql_config.mintues5_table_name)
                    min_dates = self.sql_op.get_min_date_for_codes(batch_codes, sql_config.mintues5_table_name)
                    
                    futures = []
                    for code in batch_codes:
                        current_start, current_end = start_date, end_date
                        
                        if code in max_dates and max_dates[code] is not None:
                            latest_in_db = pd.to_datetime(max_dates[code]).strftime('%Y-%m-%d')
                            earliest_in_db = pd.to_datetime(min_dates[code]).strftime('%Y-%m-%d')
                            
                            has_new = latest_in_db < end_date
                            has_old = start_date < earliest_in_db
                            
                            if has_new and has_old:
                                # Both ends missing, fetch everything to fill gaps
                                pass 
                            elif has_new:
                                current_start = latest_in_db
                            elif has_old:
                                current_end = earliest_in_db
                            else:
                                # Already up to date at both ends
                                pbar.update(1)
                                continue
                        
                        # Submit task
                        futures.append(executor.submit(fetch_data_task, code, current_start, current_end, adjustflag))

                    # Collect results
                    k_data_list = []
                    for future in as_completed(futures):
                        try:
                            res = future.result()
                            if res is not None and not res.empty:
                                k_data_list.append(res)
                        except Exception as e:
                            print(f"Task failed: {e}")
                        pbar.update(1)

                    # 2. Save batch to DB (Main Process)
                    if k_data_list:
                        all_k_data = pd.concat(k_data_list, ignore_index=True)
                        all_k_data.set_index(['code', 'time'], inplace=True)
                        self.sql_op.upsert_df_to_db(all_k_data, sql_config.mintues5_table_name, index=True)

if __name__ == '__main__':
    task = UpdateTask()
    # Example usage: now supports adjustflag='2' for forward adjustment if desired
    task.run_init_mintues5_task(start_date='2025-06-01', end_date='2026-02-11')