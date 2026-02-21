import pandas as pd
from datetime import datetime, timedelta
from typing import List, Set
from stock.tasks.base import BaseTask
from stock.database.factory import RepositoryFactory
from stock.config import settings
from stock.data_context.context import DailyContext

class IncrementalFactorUpdateTask(BaseTask):
    """
    增量更新因子 - 只更新最近N天的数据，避免全量重算
    
    原理：滚动窗口因子（ksp_sum_14d等）只需重新计算窗口期内的数据
    例如：更新最近5天的数据，实际需要查询最近5+14=19天的数据来计算
    """
    
    def __init__(self):
        super().__init__("IncrementalFactorUpdateTask")
        self.repo = RepositoryFactory.get_clickhouse_repo()
        self.max_rolling_window = 14  # 最大滚动窗口天数
    
    def run(self, lookback_days: int = 5, target_date: str = None):
        """
        执行增量因子更新
        
        Args:
            lookback_days: 需要更新的最近N天（默认5天）
            target_date: 指定目标日期（默认为昨天），格式：YYYY-MM-DD
        """
        self.log_progress(f"Starting incremental factor update (lookback={lookback_days})...")
        
        # 1. 确定需要更新的日期范围
        target_dates = self._get_target_dates(lookback_days, target_date)
        if not target_dates:
            self.log_progress("No dates to update.")
            return
        
        self.log_progress(f"Target dates to update: {target_dates}")
        
        # 2. 扩展查询范围（加上滚动窗口）
        extended_dates = self._extend_for_rolling_window(target_dates)
        self.log_progress(f"Extended query range: {len(extended_dates)} days (including {self.max_rolling_window}d rolling window)")
        
        # 3. 查询扩展范围内的数据
        df = self._query_extended_data(extended_dates)
        if df.empty:
            self.log_progress("No data found in extended range.")
            return
        
        self.log_progress(f"Loaded {len(df)} rows from {df['date'].nunique()} trading days")
        
        # 4. 计算因子（使用DailyContext的_derived_factors逻辑）
        self.log_progress("Calculating derived factors...")
        ctx = DailyContext(daily_df=df)
        df_updated = ctx.data
        
        # 5. 只保留需要更新的日期
        df_to_update = df_updated[df_updated['date'].isin(target_dates)].copy()
        
        if df_to_update.empty:
            self.log_progress("No data to update after filtering.")
            return
        
        self.log_progress(f"Updating {len(df_to_update)} rows for {len(target_dates)} days...")
        
        # 6. 准备写入数据
        df_to_save = df_to_update[DailyContext.COLUMNS].copy()
        df_to_save['volume'] = df_to_save['volume'].astype(int)
        
        # 7. 增量写入（ReplacingMergeTree会自动覆盖相同(code, date)的数据）
        chunk_size = 50000
        for i in range(0, len(df_to_save), chunk_size):
            chunk = df_to_save.iloc[i:i + chunk_size]
            self.repo.insert_df(chunk, settings.TABLE_DAILY)
            self.log_progress(f"  Inserted chunk {i//chunk_size + 1}/{(len(df_to_save)-1)//chunk_size + 1}")
        
        # 8. 优化表
        self.repo.optimize_table(settings.TABLE_DAILY)
        
        self.log_progress(f"✅ Incremental update completed: {len(df_to_update)} rows updated for {target_dates}")
    
    def _get_target_dates(self, lookback_days: int, target_date: str = None) -> List[str]:
        """获取需要更新的目标日期列表"""
        if target_date:
            # 如果指定了目标日期，以它为基础往前推
            end_date = datetime.strptime(target_date, '%Y-%m-%d')
        else:
            # 默认以昨天为结束日期
            end_date = datetime.now() - timedelta(days=1)
        
        # 获取交易日历
        trade_dates_df = self.repo.query(f"""
            SELECT DISTINCT date FROM {settings.TABLE_DAILY}
            WHERE date <= '{end_date.strftime('%Y-%m-%d')}'
            ORDER BY date DESC
            LIMIT {lookback_days}
        """)
        
        if trade_dates_df.empty:
            return []
        
        # 按正序排列（从旧到新）
        target_dates = sorted(trade_dates_df['date'].tolist())
        return target_dates
    
    def _extend_for_rolling_window(self, target_dates: List[str]) -> List[str]:
        """扩展查询范围以包含滚动窗口所需的历史数据"""
        if not target_dates:
            return []
        
        earliest_target = min(target_dates)
        latest_target = max(target_dates)
        
        # 需要额外查询的最早日期
        earliest_needed = (datetime.strptime(earliest_target, '%Y-%m-%d') - 
                          timedelta(days=self.max_rolling_window * 2)).strftime('%Y-%m-%d')
        
        # 查询扩展范围内的所有交易日
        extended_df = self.repo.query(f"""
            SELECT DISTINCT date FROM {settings.TABLE_DAILY}
            WHERE date >= '{earliest_needed}' AND date <= '{latest_target}'
            ORDER BY date ASC
        """)
        
        return extended_df['date'].tolist()
    
    def _query_extended_data(self, extended_dates: List[str]) -> pd.DataFrame:
        """查询扩展范围内的所有数据"""
        if not extended_dates:
            return pd.DataFrame()
        
        # 将日期列表转换为SQL IN子句格式
        date_list = "'" + "','".join(extended_dates) + "'"
        
        query = f"""
            SELECT * FROM {settings.TABLE_DAILY}
            WHERE date IN ({date_list})
            ORDER BY date ASC
        """
        
        return self.repo.query(query)
    
    def close(self):
        self.repo.close()
