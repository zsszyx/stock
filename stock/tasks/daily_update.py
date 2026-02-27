import pandas as pd
from typing import List, Optional
from stock.tasks.base import BaseTask
from stock.database.factory import RepositoryFactory
from stock.config import settings
from stock.data_context.context import DailyContext

class DailyAggregationTask(BaseTask):
    """
    解耦后的日线聚合任务：
    职责：读取 ClickHouse 中的分钟线 -> 聚合计算因子 -> 写入日线表
    """
    def __init__(self, chunk_size: int = 5):
        super().__init__("DailyAggregationTask")
        self.chunk_size = chunk_size
        self.repo = RepositoryFactory.get_clickhouse_repo()

    def run(self, start_date: str, end_date: str, clear_target: bool = False):
        self.log_progress(f"🚀 开始聚合任务: {start_date} -> {end_date} (清理目标={clear_target})")
        
        # ReplacingMergeTree naturally handles duplicates on (code, date). 
        # Manual DELETE is risky due to asynchronicity.
        
        # 1. 获取分钟线表中存在的日期
        query_dates = f"SELECT DISTINCT date FROM {settings.TABLE_MIN5} WHERE date >= '{start_date}' AND date <= '{end_date}' ORDER BY date ASC"
        available_dates = self.repo.query(query_dates)['date'].tolist()
        
        if not available_dates:
            self.log_progress("⚠️  未在 5 分钟线表中找到指定范围的数据。")
            return

        self.log_progress(f"📊 发现 {len(available_dates)} 天待处理数据，开始分块处理...")

        for i in range(0, len(available_dates), self.chunk_size):
            chunk = available_dates[i : i + self.chunk_size]
            self._process_chunk(chunk)

        # 2. 物理去重优化
        self.repo.optimize_table(settings.TABLE_DAILY)
        self.log_progress("🏁 日线聚合与因子计算完成。")
        
        # 3. 自动健康自检
        from stock.utils.health_check import DataHealthMonitor
        monitor = DataHealthMonitor(repo=self.repo)
        monitor.validate_or_raise()

    def _process_chunk(self, dates: List[str]):
        try:
            start, end = dates[0], dates[-1]
            # 从分钟线表读取已补全的数据 (只读取必要列以减少内存和IO消耗)
            cols = "date, time, code, open, high, low, close, volume, amount"
            query = f"SELECT {cols} FROM {settings.TABLE_MIN5} WHERE date >= '{start}' AND date <= '{end}'"
            df_min5 = self.repo.query(query)
            
            if df_min5.empty: return
                
            # 利用优化后的 DailyContext 进行高性能聚合
            df_daily = DailyContext.from_min5(df_min5)
            
            # 写入日线表
            self.repo.insert_df(df_daily, settings.TABLE_DAILY)
            self.log_progress(f"  ✅ 已完成块: {start} 至 {end} ({len(df_daily)} 行)")
        except Exception as e:
            self.log_error(f"❌ 聚合块 {dates[0]} 失败", e)

    def close(self):
        self.repo.close()
