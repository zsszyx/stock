import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional
from stock.selector.concept_selector import BaseSelector
from stock.data_context.context import DailyContext

class WyckoffSelector(BaseSelector):
    """
    威科夫选股策略 (Wyckoff Breakout Hunter)
    
    逻辑：
    1. 吸筹识别 (Accumulation): 寻找过去 60 个交易日波动率 (Volatility) 最低的前 30% 股票。
       - 波动率定义: 60日收盘价标准差 / 60日收盘价均值
    2. 启动识别 (Launch): 在低波动池中，选取近期动能 (5日 KSP 累计分) 最强的股票。
    3. 过滤: 上市满 180 天。
    """
    def __init__(self, daily_context: DailyContext, vol_window: int = 60, vol_percentile: float = 0.3):
        super().__init__(daily_context)
        self.vol_window = vol_window
        self.vol_percentile = vol_percentile

    def select(self, date: datetime, candidate_codes: Optional[List[str]] = None, **kwargs) -> List[str]:
        # 获取足够计算波动率的历史窗口
        df = self.context.get_window(date, window_days=self.vol_window + 1, codes=candidate_codes)
        if df.empty: return []
        
        target_date_str = date.strftime('%Y-%m-%d')
        
        # 1. 基础过滤：仅处理当天存在的股票
        # 优化：先筛选出当天有数据的股票代码，再回溯它们的历史
        today_data = df[df['date'] == target_date_str]
        valid_codes = today_data['code'].unique()
        
        if len(valid_codes) == 0: return []
        
        # 提取这些股票的历史数据进行计算
        hist_df = df[df['code'].isin(valid_codes)].copy()
        hist_df = hist_df.sort_values(['code', 'date'])
        
        # 2. 计算波动率 (Volatility)
        # Volatility = Rolling StdDev / Rolling Mean
        hist_df['std'] = hist_df.groupby('code')['close'].transform(lambda x: x.rolling(self.vol_window).std())
        hist_df['mean'] = hist_df.groupby('code')['close'].transform(lambda x: x.rolling(self.vol_window).mean())
        hist_df['volatility'] = hist_df['std'] / hist_df['mean']
        
        # 再次提取当日数据（此时已包含计算好的 volatility）
        today_df = hist_df[hist_df['date'] == target_date_str].copy()
        
        # 3. 过滤：上市满 180 天 且 日成交额 >= 5000万
        if 'list_days' in today_df.columns:
            today_df = today_df[today_df['list_days'] >= 180]
            
        if 'amount' in today_df.columns:
            # 过滤成交额小于 5000万 的股票 (单位为元)
            today_df = today_df[today_df['amount'] >= 50000000]
            
        if today_df.empty: return []
        
        # 4. 识别吸筹：低波动
        # 选取波动率最小的前 30%
        # dropna() 是为了防止数据不足 60 天导致的 NaN
        clean_df = today_df.dropna(subset=['volatility'])
        if clean_df.empty: return []
        
        vol_threshold = clean_df['volatility'].quantile(self.vol_percentile)
        accumulation_pool = clean_df[clean_df['volatility'] <= vol_threshold].copy()
        
        # 5. 识别启动：高 KSP 动能
        # 在低波动池中，选 KSP 5日分最高的
        if 'ksp_sum_5d' in accumulation_pool.columns:
            # 优先选正向动能
            accumulation_pool = accumulation_pool.sort_values('ksp_sum_5d', ascending=False)
        else:
            # Fallback
            accumulation_pool = accumulation_pool.sort_values('pct_chg', ascending=False)
            
        # 返回前 9 名
        return accumulation_pool.head(9)['code'].tolist()
