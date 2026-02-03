import sys
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bao.bao_interface import BaoInterface
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

def fetch_data_task(code, start_date, end_date):
    """
    Module-level task function for multiprocessing.
    Uses the global worker_bi instance.
    """
    global worker_bi
    try:
        if worker_bi is None:
            # Fallback if init didn't run (shouldn't happen with correct Executor setup)
            init_worker()
        return worker_bi.get_k_data_5min(code=code, start_date=start_date, end_date=end_date)
    except Exception as e:
        # In multiprocessing, print might not show up immediately or at all depending on config
        # but it's better than silent failure.
        return None

class UpdateTask:
    def __init__(self):
        self.sql_op = SqlOp()

    def run_init_mintues5_task(self, start_date: str, end_date: str, max_workers: int = 6, batch_size: int = 50):
        """
        Initializes/Updates 5-minute K-line data using Multiprocessing.
        
        Args:
            start_date: Default start date if no data exists.
            end_date: End date for data fetching.
            max_workers: Number of parallel processes. (Baostock isn't thread-safe, so we use processes)
            batch_size: Number of stocks to process and save in one transaction.
        """
        # Note: Reduced default max_workers to 6 to be conservative with system resources
        print(f"Starting update task with {max_workers} processes, batch size {batch_size}...")
        
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
            # This excludes indices (sh.000, sz.399), STAR market (sh.688), Beijing (bj.8), Funds (51, 15), etc.
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
            # --- END NEW FILTER ---
            
            all_codes = stock_list['code'].tolist()
            print(f"Total stocks to process: {len(all_codes)}")
        
        # Use ProcessPoolExecutor for BaoStock interactions
        # The 'initializer' ensures we login once per process
        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
            with tqdm(total=len(all_codes), desc="Updating Stocks") as pbar:
                for i in range(0, len(all_codes), batch_size):
                    batch_codes = all_codes[i:i+batch_size]
                    
                    # 1. Get existing max dates for this batch (Main Process DB Check)
                    max_dates = self.sql_op.get_max_date_for_codes(batch_codes, sql_config.mintues5_table_name)
                    
                    futures = []
                    for code in batch_codes:
                        current_start = start_date
                        
                        # Incremental update logic
                        if code in max_dates and max_dates[code] is not None:
                            latest_date_in_db = pd.to_datetime(max_dates[code])
                            if latest_date_in_db.strftime('%Y-%m-%d') >= last_day:
                                pbar.update(1)
                                continue
                            current_start = latest_date_in_db.strftime('%Y-%m-%d')
                        
                        # Submit task to process pool
                        futures.append(executor.submit(fetch_data_task, code, current_start, end_date))

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
    # Example usage
    task.run_init_mintues5_task(start_date='2025-12-01', end_date='2026-02-02')