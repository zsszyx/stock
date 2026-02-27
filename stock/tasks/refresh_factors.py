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
        
        # 显式转换所有整数列，防止 CSV 导出时出现 .0 导致 ClickHouse 解析失败
        int_cols = ['volume', 'ksp_rank', 'ksp_sum_14d_rank', 'ksp_sum_10d_rank', 'ksp_sum_7d_rank', 'ksp_sum_5d_rank', 'list_days', 'is_listed_180']
        for col in int_cols:
            if col in df_to_save.columns:
                df_to_save[col] = df_to_save[col].fillna(0).astype(int)
        
        # 分块插入以确保 ClickHouse 处理平稳
        chunk_size = 50000
        for i in range(0, len(df_to_save), chunk_size):
            chunk = df_to_save.iloc[i : i + chunk_size]
            self.repo.insert_df(chunk, settings.TABLE_DAILY)
            
        self.repo.optimize_table(settings.TABLE_DAILY)
        self.log_progress("Factor refresh completed successfully.")
        
        # 3. 自动健康自检
        from stock.utils.health_check import DataHealthMonitor
        monitor = DataHealthMonitor(repo=self.repo)
        monitor.validate_or_raise()

    def close(self):
        self.repo.close()
