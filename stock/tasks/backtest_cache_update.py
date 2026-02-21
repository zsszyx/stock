import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Set, Optional
from stock.tasks.base import BaseTask
from stock.database.factory import RepositoryFactory
from stock.config import settings
from stock.data_context.context import DailyContext
from stock.selector import KSPScoreSelector, ConceptSelector
from stock.data_context.concept_context_v2 import ConceptContext

class BacktestCacheUpdateTask(BaseTask):
    """
    回测数据缓存维护任务
    
    预计算内容：
    1. 历史候选股对齐数据（时间对齐、前向填充）
    2. 只缓存曾出现在选股信号中的股票（减少数据量）
    
    使用场景：
    - 每日收盘后运行，缓存当日数据
    - 全量重建时使用（数据修正后）
    """
    
    def __init__(self):
        super().__init__("BacktestCacheUpdateTask")
        self.repo = RepositoryFactory.get_clickhouse_repo()
        self.cache_table = "backtest_feed_cache"
    
    def run(self, start_date: str = None, end_date: str = None, full_rebuild: bool = False):
        """
        执行缓存更新
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)，默认从今天往前推60天
            end_date: 结束日期，默认今天
            full_rebuild: 是否全量重建（删除旧缓存）
        """
        # 确定日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        self.log_progress(f"Starting backtest cache update: {start_date} to {end_date}")
        
        # 确保缓存表存在
        self._ensure_cache_table_exists()
        
        if full_rebuild:
            self.log_progress("Full rebuild mode: clearing old cache...")
            self.repo.execute(f"TRUNCATE TABLE {self.cache_table}")
        
        # 1. 获取历史候选股列表（曾出现在选股信号中的股票）
        self.log_progress("Identifying historical candidate stocks...")
        candidate_codes = self._get_historical_candidates(start_date, end_date)
        
        if not candidate_codes:
            self.log_progress("No candidate stocks found.")
            return
        
        self.log_progress(f"Found {len(candidate_codes)} unique candidate stocks")
        
        # 2. 获取交易日历（用于对齐）
        trading_dates = self._get_trading_dates(start_date, end_date)
        if not trading_dates:
            self.log_progress("No trading dates found.")
            return
        
        self.log_progress(f"Trading dates: {len(trading_dates)} days")
        full_idx = pd.to_datetime(trading_dates)
        
        # 3. 批量处理股票数据
        self.log_progress("Processing and aligning stock data...")
        total_processed = 0
        batch_size = 50  # 每批处理50只股票
        
        for i in range(0, len(candidate_codes), batch_size):
            batch_codes = list(candidate_codes)[i:i + batch_size]
            self._process_batch(batch_codes, full_idx, trading_dates)
            total_processed += len(batch_codes)
            
            if (i // batch_size + 1) % 10 == 0:
                self.log_progress(f"  Progress: {total_processed}/{len(candidate_codes)} stocks")
        
        # 4. 优化表
        self.log_progress(f"Optimizing cache table...")
        self.repo.optimize_table(self.cache_table)
        
        self.log_progress(f"✅ Cache update completed: {total_processed} stocks cached")
        
        # 5. 报告统计
        stats = self._get_cache_stats()
        self.log_progress(f"Cache stats: {stats['total_stocks']} stocks, {stats['total_rows']} rows, "
                         f"date range: {stats['min_date']} to {stats['max_date']}")
    
    def _get_historical_candidates(self, start_date: str, end_date: str) -> Set[str]:
        """获取历史候选股（曾出现在选股信号中的股票）"""
        # 使用概念选股策略模拟历史选股结果
        daily_df = self.repo.query(f"""
            SELECT * FROM {settings.TABLE_DAILY}
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date ASC
        """)
        
        if daily_df.empty:
            return set()
        
        # 初始化选股器
        daily_ctx = DailyContext(daily_df=daily_df)
        concept_ctx = ConceptContext(repo=self.repo)
        score_selector = KSPScoreSelector(daily_ctx, ksp_period=5)
        strategy_obj = ConceptSelector(
            score_selector, concept_ctx,
            top_concepts_n=3,
            top_stocks_per_concept_n=3,
            min_amount_filter=50000000
        )
        
        # 收集所有出现过的候选股
        candidate_codes = set()
        trading_dates = sorted(daily_df['date'].unique())
        
        for date_str in trading_dates:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            selected = strategy_obj.select(dt)
            candidate_codes.update(selected)
        
        return candidate_codes
    
    def _get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日历"""
        df = self.repo.query(f"""
            SELECT DISTINCT date FROM {settings.TABLE_DAILY}
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date ASC
        """)
        return df['date'].tolist() if not df.empty else []
    
    def _process_batch(self, codes: List[str], full_idx: pd.DatetimeIndex, trading_dates: List[str]):
        """处理一批股票数据并写入缓存"""
        # 查询这批股票的所有数据
        code_list = "'" + "','".join(codes) + "'"
        df_all = self.repo.query(f"""
            SELECT * FROM {settings.TABLE_DAILY}
            WHERE code IN ({code_list}) AND date IN ('{"','".join(trading_dates)}')
            ORDER BY code, date ASC
        """)
        
        if df_all.empty:
            return
        
        # 为每只股票进行对齐处理
        cache_rows = []
        
        for code in codes:
            code_df = df_all[df_all['code'] == code].copy()
            if code_df.empty:
                continue
            
            # 设置datetime索引
            code_df['datetime'] = pd.to_datetime(code_df['date'])
            code_df = code_df.set_index('datetime').sort_index()
            
            # 对齐到完整交易日历
            aligned_df = code_df.reindex(full_idx)
            
            # 填充OHLC和因子数据
            fill_cols = ['open', 'high', 'low', 'close', 'poc', 
                        'ksp_sum_5d', 'ksp_sum_5d_rank', 'ksp_sum_7d', 'ksp_sum_7d_rank',
                        'ksp_sum_14d', 'ksp_sum_14d_rank', 'ksp_score', 'ksp_rank',
                        'list_days', 'amount']
            
            for col in fill_cols:
                if col in aligned_df.columns:
                    if col in ['open', 'high', 'low', 'close', 'poc']:
                        # 价格数据：小于0.01的设为NaN，然后前向填充
                        aligned_df.loc[aligned_df[col] <= 0.01, col] = np.nan
                    aligned_df[col] = aligned_df[col].ffill()
            
            # volume填0（表示停牌）
            aligned_df['volume'] = aligned_df['volume'].fillna(0).astype(int)
            
            # 确保list_days存在
            if 'list_days' not in aligned_df.columns:
                aligned_df['list_days'] = 0
            aligned_df['list_days'] = aligned_df['list_days'].fillna(0).astype(int)
            
            # 准备缓存行
            aligned_df = aligned_df.reset_index()
            aligned_df['code'] = code
            # 重置索引后，原索引列名为 'index'，需要重命名为 'datetime'
            if 'index' in aligned_df.columns:
                aligned_df = aligned_df.rename(columns={'index': 'datetime'})
            aligned_df['date'] = aligned_df['datetime'].dt.strftime('%Y-%m-%d')
            
            # 选择需要的列
            cache_cols = ['code', 'date', 'datetime', 'open', 'high', 'low', 'close', 'volume',
                         'ksp_sum_5d', 'ksp_sum_5d_rank', 'ksp_sum_7d', 'ksp_sum_7d_rank',
                         'ksp_sum_14d', 'ksp_sum_14d_rank', 'ksp_score', 'ksp_rank',
                         'poc', 'list_days', 'amount']
            
            available_cols = [c for c in cache_cols if c in aligned_df.columns]
            cache_df = aligned_df[available_cols].copy()
            
            cache_rows.append(cache_df)
        
        # 批量写入缓存表
        if cache_rows:
            combined_df = pd.concat(cache_rows, ignore_index=True)
            self.repo.insert_df(combined_df, self.cache_table)
    
    def _ensure_cache_table_exists(self):
        """确保缓存表存在"""
        # 检查表是否存在
        result = self.repo.query(f"""
            SHOW TABLES LIKE '{self.cache_table}'
        """)
        
        if result.empty:
            self.log_progress(f"Creating cache table: {self.cache_table}")
            # 读取SQL文件并执行
            with open('scripts/create_backtest_cache_table.sql', 'r') as f:
                sql = f.read()
            self.repo.execute(sql)
    
    def _get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        stats = self.repo.query(f"""
            SELECT 
                count() as total_rows,
                uniq(code) as total_stocks,
                min(date) as min_date,
                max(date) as max_date
            FROM {self.cache_table}
        """)
        
        if stats.empty:
            return {'total_rows': 0, 'total_stocks': 0, 'min_date': None, 'max_date': None}
        
        return {
            'total_rows': stats.iloc[0]['total_rows'],
            'total_stocks': stats.iloc[0]['total_stocks'],
            'min_date': stats.iloc[0]['min_date'],
            'max_date': stats.iloc[0]['max_date']
        }
    
    def close(self):
        self.repo.close()
