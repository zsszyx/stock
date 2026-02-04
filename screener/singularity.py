from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from stock.screener.base import BaseScreener
from stock.data.context import MarketContext
from stock.factors.distribution_analyzer import DistributionAnalyzer

class SingularityScreener(BaseScreener):
    """
    实现奇异点选股逻辑。
    过滤条件: Skew < threshold
    排序条件: Kurtosis 降序
    """
    
    def __init__(self, skew_threshold: float = -0.0, top_n: int = 5, filter_candidates: bool = True):
        super().__init__(filter_candidates=filter_candidates)
        self.skew_threshold = skew_threshold
        self.top_n = top_n
        
    def scan(self, context: MarketContext) -> pd.DataFrame:
        """
        利用五分钟上下文获取当日数据。
        """
        # 【关键修复】：显式遵循“只计算上一环节存留股票”的逻辑
        # 如果开启了 filter_candidates，则 codes 为上一环节选出的结果；否则为 None (计算全量)
        codes = context.candidate_codes if self.filter_candidates else None
        
        # 显式传递 codes 给 get_window，确保内存切片时只包含所需股票
        daily_df = context.minutes5.get_window(
            date=context.current_date, 
            window_days=1, 
            codes=codes
        )
        
        if daily_df.empty:
            return pd.DataFrame()
            
        # 2. 计算因子 (只针对 daily_df 中的股票)
        results_df = self._calculate_factors(daily_df)
            
        if results_df.empty:
            return pd.DataFrame()

        # 3. 过滤 (Skew)
        filtered = results_df[results_df['skew'] < self.skew_threshold].copy()
        
        # 4. 排序 (Kurt)
        ranked = filtered.sort_values(by='kurt', ascending=False)
        
        # 5. 返回 Top N
        if self.top_n:
            return ranked.head(self.top_n)
        
        return ranked

    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """从原始分钟数据计算统计因子"""
        results = []
        grouped = df.groupby('code')
        
        for code, group in grouped:
            valid_group = group[group['volume'] > 0]
            if valid_group.empty:
                continue
                
            prices = (valid_group['amount'] / valid_group['volume']).tolist()
            volumes = valid_group['volume'].tolist()
            
            analyzer = DistributionAnalyzer(prices, volumes)
            if not analyzer.is_valid:
                continue
            
            results.append({
                'code': code,
                'skew': analyzer.skewness,
                'kurt': analyzer.kurtosis,
                'poc': analyzer.poc,
                'close': valid_group['close'].iloc[-1]
            })
            
        return pd.DataFrame(results)
