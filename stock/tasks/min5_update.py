import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Optional

from stock.tasks.base import BaseTask
from stock.database.factory import RepositoryFactory
from stock.data_fetch.data_provider.baostock_provider import BaoInterface
from stock.config import settings

def fetch_data_task(code, start_date, end_date, adjustflag='3'):
    """Module-level task function for multiprocessing."""
    try:
        bi = BaoInterface()
        return bi.get_k_data_5min(code=code, start_date=start_date, end_date=end_date, adjustflag=adjustflag)
    except Exception:
        return None

class Min5UpdateTask(BaseTask):
    def __init__(self, max_workers: int = 6, batch_size: int = 50):
        super().__init__("Min5UpdateTask")
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.repo = RepositoryFactory.get_clickhouse_repo()

    def run(self, start_date: str, end_date: str, adjustflag: str = "3", full_update: bool = False):
        self.log_progress(f"Starting update: {start_date} to {end_date} (adjust={adjustflag}, full={full_update})")
        
        # Ensure ClickHouse table exists
        self.repo.create_mintues5_table()
        
        # Get stock list using BaoInterface
        with BaoInterface() as bi:
            trade_dates = bi.get_trade_dates(start_date=start_date, end_date=end_date)
            if trade_dates is None or trade_dates.empty:
                raise ValueError(f"Fail Fast: No trade dates found in range {start_date} to {end_date}")
            
            last_day = trade_dates.iloc[-1]['calendar_date']
            stock_list = bi.get_stock_list(date=last_day)
            if stock_list is None or stock_list.empty:
                raise ConnectionError(f"Fail Fast: Failed to retrieve stock list from provider for date {last_day}")

            # Filter Main Board and ChiNext, exclude ST
            stock_list = stock_list[stock_list['code'].str.match(r'^(sh\.60|sz\.00|sz\.30)')]
            
            # Use non-regex contains for safety
            is_st = stock_list['code_name'].str.contains('ST', case=False, na=False, regex=False)
            is_s = stock_list['code_name'].str.contains('S', case=False, na=False, regex=False)
            is_t = stock_list['code_name'].str.contains('T', case=False, na=False, regex=False)
            
            stock_list = stock_list[~(is_st | is_s | is_t)]
            all_codes = stock_list['code'].tolist()
            self.log_progress(f"Total stocks in scope: {len(all_codes)}")

        with ProcessPoolExecutor(max_workers=self.max_workers, initializer=BaoInterface.worker_init) as executor:
            with tqdm(total=len(all_codes), desc="Updating Min5") as pbar:
                for i in range(0, len(all_codes), self.batch_size):
                    batch_codes = all_codes[i:i+self.batch_size]
                    
                    max_dates = {}
                    min_dates = {}
                    if not full_update:
                        max_dates = self.repo.get_max_date_for_codes(batch_codes, settings.TABLE_MIN5)
                        min_dates = self.repo.get_min_date_for_codes(batch_codes, settings.TABLE_MIN5)

                    futures = []
                    for code in batch_codes:
                        fetch_start, fetch_end = start_date, end_date
                        if not full_update and code in max_dates:
                            latest_in_db = max_dates[code]
                            earliest_in_db = min_dates[code]
                            has_new = latest_in_db < end_date
                            has_old = start_date < earliest_in_db
                            if has_new and has_old: pass 
                            elif has_new: fetch_start = latest_in_db
                            elif has_old: fetch_end = earliest_in_db
                            else:
                                pbar.update(1)
                                continue
                        futures.append(executor.submit(fetch_data_task, code, fetch_start, fetch_end, adjustflag))

                    k_data_list = []
                    for future in as_completed(futures):
                        try:
                            res = future.result()
                            if res is not None and not res.empty:
                                k_data_list.append(res)
                        except Exception as e:
                            self.logger.error(f"Task failed: {e}")
                        pbar.update(1)

                    if k_data_list:
                        all_k_data = pd.concat(k_data_list, ignore_index=True)
                        all_k_data['volume'] = all_k_data['volume'].astype(int)
                        self.repo.insert_df(all_k_data, settings.TABLE_MIN5)
        
        self.log_progress(f"Optimizing table {settings.TABLE_MIN5}...")
        self.repo.optimize_table(settings.TABLE_MIN5)
        self.log_progress("Min5 update and optimization completed.")

    def close(self):
        self.repo.close()
