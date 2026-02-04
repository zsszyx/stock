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
        # 显式遵循“只计算上一环节存留股票”的逻辑
        codes = context.candidate_codes if self.filter_candidates else None

        # 显式声明只需要当日 1 天的数据
        # 使用 context.current_date 确保时间正确
        daily_df = context.minutes5.get_window(date=context.current_date, window_days=1, codes=codes)
        
        if daily_df.empty:
            return pd.DataFrame()
            
        # 2. 计算因子
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
                
            # 使用统一的工厂方法计算均价分布
            analyzer = DistributionAnalyzer.from_amount_volume(
                amounts=valid_group['amount'].values,
                volumes=valid_group['volume'].values
            )
            
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
