import pandas as pd
import numpy as np
from typing import List

class KSPFactorEngine:
    """
    KSP 因子生成引擎：封装所有统计学因子计算逻辑
    """
    @staticmethod
    def calculate_ksp_scores(df: pd.DataFrame) -> pd.DataFrame:
        """计算基础 KSP 分数"""
        magnitude = df['kurt'].abs() * df['skew'].abs()
        is_bad = (df['kurt'] < 0) | (df['skew'] > 0)
        df['ksp_score'] = np.where(is_bad, -magnitude, magnitude)
        return df

    @staticmethod
    def add_rolling_factors(df: pd.DataFrame, periods: List[int] = [5, 7, 10, 14]) -> pd.DataFrame:
        """添加滚动累计因子及排名"""
        df = df.sort_values(['code', 'date'])
        for p in periods:
            col_name = f'ksp_sum_{p}d'
            df[col_name] = df.groupby('code')['ksp_score'].transform(
                lambda x: x.rolling(window=p, min_periods=1).sum()
            )
            # 每日排名
            df[f'{col_name}_rank'] = df.groupby('date')[col_name].rank(ascending=False, method='min')
        
        df['ksp_rank'] = df.groupby('date')['ksp_score'].rank(ascending=False, method='min')
        return df

    @staticmethod
    def add_price_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """添加价格衍生指标"""
        df = df.sort_values(['code', 'date'])
        df['prev_close'] = df.groupby('code')['close'].shift(1)
        df['pct_chg'] = (df['close'] - df['prev_close']) / df['prev_close']
        df['amplitude'] = (df['high'] - df['low']) / df['prev_close']
        return df
