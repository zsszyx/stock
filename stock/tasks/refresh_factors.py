import pandas as pd
from stock.tasks.base import BaseTask
from stock.database.factory import RepositoryFactory
from stock.config import settings
from stock.data_context.context import DailyContext

class RefreshFactorsTask(BaseTask):
    def __init__(self):
        super().__init__("RefreshFactorsTask")
        self.repo = RepositoryFactory.get_clickhouse_repo()

    def run(self):
        self.log_progress(f"Reading all daily data from {settings.TABLE_DAILY}...")
        df = self.repo.query(f"SELECT * FROM {settings.TABLE_DAILY} ORDER BY date ASC")
        
        if df.empty:
            self.log_progress("Daily table is empty. Nothing to refresh.")
            return

        self.log_progress(f"Calculating rolling factors for {len(df)} rows...")
        
        # 1. 使用 DailyContext 的逻辑计算衍生因子 (包含 ksp_score 和 ksp_sum_14d)
        # 注意：DailyContext.__init__ 会自动调用 _add_derived_factors
        ctx = DailyContext(daily_df=df)
        df_updated = ctx.data
        
        # 2. 将计算结果写回 ClickHouse
        # 既然我们使用了 ReplacingMergeTree，相同的 (code, date) 会被更新
        self.log_progress("Writing updated factors back to ClickHouse...")
        
        # 转换为正确的类型
        df_to_save = df_updated[DailyContext.COLUMNS].copy()
        df_to_save['volume'] = df_to_save['volume'].astype(int)
        
        # 分块插入以确保 ClickHouse 处理平稳
        chunk_size = 50000
        for i in range(0, len(df_to_save), chunk_size):
            chunk = df_to_save.iloc[i : i + chunk_size]
            self.repo.insert_df(chunk, settings.TABLE_DAILY)
            
        self.repo.optimize_table(settings.TABLE_DAILY)
        self.log_progress("Factor refresh completed successfully.")

    def close(self):
        self.repo.close()
