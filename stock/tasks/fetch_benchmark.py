import pandas as pd
from stock.tasks.base import BaseTask
from stock.database.factory import RepositoryFactory
from stock.data_fetch.data_provider.baostock_provider import BaoInterface
from stock.config import settings

class FetchBenchmarkTask(BaseTask):
    def __init__(self):
        super().__init__("FetchBenchmarkTask")
        self.repo = RepositoryFactory.get_clickhouse_repo()

    def run(self, start_date: str = "2025-01-01", end_date: str = None):
        if end_date is None:
            from datetime import datetime
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        self.log_progress(f"Fetching SSE Index (sh.000001) from {start_date} to {end_date}...")
        
        self.repo.create_benchmark_table()
        
        with BaoInterface() as bi:
            # sh.000001 是上证指数在 BaoStock 中的代码
            df = bi.get_k_data_daily(code="sh.000001", start_date=start_date, end_date=end_date)
            
            if df is not None and not df.empty:
                # 转换类型以匹配 ClickHouse Schema
                df['volume'] = df['volume'].astype(float).astype(int)
                df['amount'] = df['amount'].astype(float)
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = df[col].astype(float)
                
                self.repo.insert_df(df, settings.TABLE_BENCHMARK)
                self.repo.optimize_table(settings.TABLE_BENCHMARK)
                self.log_progress(f"Successfully saved {len(df)} days of benchmark data.")
            else:
                self.log_error("Failed to fetch benchmark data.", ValueError("Empty result from provider"))

    def close(self):
        self.repo.close()
