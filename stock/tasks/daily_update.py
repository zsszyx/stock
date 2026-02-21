import pandas as pd
from typing import List, Optional
from stock.tasks.base import BaseTask
from stock.database.factory import RepositoryFactory
from stock.config import settings
from stock.data_context.context import DailyContext

class DailyUpdateTask(BaseTask):
    def __init__(self, chunk_size: int = None):
        super().__init__("DailyUpdateTask")
        self.chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        self.repo = RepositoryFactory.get_clickhouse_repo()

    def run(self, full_recompute: bool = False):
        self.log_progress(f"Starting update (full={full_recompute})")
        
        if full_recompute:
            self.log_progress(f"Re-creating table {settings.TABLE_DAILY} for new schema...")
            self.repo.execute(f"DROP TABLE IF EXISTS {settings.TABLE_DAILY}")
        
        self.repo.create_daily_kline_table()
        
        # 获取所有待处理日期
        all_min5_dates = self.repo.get_all_dates(settings.TABLE_MIN5)
        
        if not full_recompute:
            res = self.repo.query(f"SELECT DISTINCT date FROM {settings.TABLE_DAILY}")
            existing_dates = set(res['date'].tolist()) if not res.empty else set()
            dates_to_update = [d for d in all_min5_dates if d not in existing_dates]
        else:
            dates_to_update = all_min5_dates

        if not dates_to_update:
            self.log_progress("Already up to date.")
            return

        self.log_progress(f"Updating {len(dates_to_update)} dates in chunks of {self.chunk_size}...")
        
        for i in range(0, len(dates_to_update), self.chunk_size):
            chunk = dates_to_update[i : i + self.chunk_size]
            self._process_chunk(chunk)

        # 物理去重
        self.log_progress(f"Optimizing table {settings.TABLE_DAILY}...")
        self.repo.optimize_table(settings.TABLE_DAILY)
        self.log_progress("Daily table update and optimization completed.")

    def _process_chunk(self, dates: List[str]):
        try:
            start, end = dates[0], dates[-1]
            query = f"SELECT * FROM {settings.TABLE_MIN5} WHERE date >= '{start}' AND date <= '{end}'"
            df_min5 = self.repo.query(query)
            
            if df_min5 is None or df_min5.empty:
                return
                
            # 这里调用 from_min5
            # 注意：在分块模式下，ksp_sum_14d 的初始几天可能不准，
            # 我们将在下一步的 ComputeFactorsTask 中全量重刷日线表的因子
            df_daily = DailyContext.from_min5(df_min5)
            df_to_insert = df_daily[DailyContext.COLUMNS].copy()
            df_to_insert['volume'] = df_to_insert['volume'].astype(int)
            
            self.repo.insert_df(df_to_insert, settings.TABLE_DAILY)
            self.log_progress(f"  Processed {start} to {end}")
        except Exception as e:
            self.log_error(f"Failed chunk {dates[0]}", e)

    def close(self):
        self.repo.close()
