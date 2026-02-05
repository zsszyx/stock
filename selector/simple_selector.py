import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
from data_context.context import DailyContext

class RecentDaysPctChgSelector:
    """
    选股器：选择最近 N 日涨跌幅在指定范围内的股票。
    """
    def __init__(self, daily_context: DailyContext):
        self.context = daily_context

    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None, 
               days: int = 5, 
               threshold: float = 0.05) -> List[str]:
        """
        选择以传入日期为准，最近 days 日涨跌幅绝对值不超过 threshold 的股票。
        
        Args:
            date: 目标日期
            candidate_codes: 候选股票代码列表，若为 None 则从 context 中所有股票选择
            days: 统计的天数
            threshold: 涨跌幅阈值（绝对值）
            
        Returns:
            符合条件的股票 code 列表
        """
        # 获取最近 N 天的数据
        # get_window 返回的是包含 date 在内的最近 N 天数据
        df = self.context.get_window(date, window_days=days, codes=candidate_codes)
        
        if df.empty:
            return []

        # 计算累积涨跌幅
        # 这里的 pct_chg 是 (close - prev_close) / prev_close
        # 总涨跌幅 = (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
        def calc_total_ret(group):
            # 过滤掉 NaN（第一天可能没有 prev_close）
            returns = group['pct_chg'].dropna()
            if len(returns) == 0:
                return np.nan
            total_ret = (1 + returns).prod() - 1
            return total_ret

        # 按照 code 分组计算总涨跌幅
        # 注意：如果某只股票在最近 days 天内交易天数不足，也会被计算
        # 如果需要严格要求必须满 days 天，可以增加判断
        stats = df.groupby('code').apply(calc_total_ret, include_groups=False)
        
        # 筛选绝对值不超过 threshold 的股票
        selected = stats[stats.abs() <= threshold]
        
        return selected.index.tolist()

class MaxDailyPctChgSelector:
    """
    选股器：选择最近 N 日内，每一天的涨跌幅绝对值都不超过指定阈值的股票。
    """
    def __init__(self, daily_context: DailyContext):
        self.context = daily_context

    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None, 
               days: int = 5, 
               threshold: float = 0.05) -> List[str]:
        """
        选择以传入日期为准，最近 days 日内每日涨跌幅绝对值最大值不超过 threshold 的股票。
        
        Args:
            date: 目标日期
            candidate_codes: 候选股票代码列表
            days: 统计的天数
            threshold: 每日涨跌幅阈值（绝对值）
            
        Returns:
            符合条件的股票 code 列表
        """
        # 获取最近 N 天的数据
        df = self.context.get_window(date, window_days=days, codes=candidate_codes)
        
        if df.empty:
            return []

        # 计算每只股票最近 N 天涨跌幅绝对值的最大值
        # 注意：pct_chg 可能包含 NaN（如果数据起始日没有前收盘价），dropna 排除之
        stats = df.groupby('code')['pct_chg'].apply(lambda x: x.abs().max())
        
        # 筛选最大绝对涨跌幅不超过 threshold 的股票
        selected = stats[stats <= threshold]
        
        return selected.index.tolist()

class POCNearSelector:
    """
    选股器：选择收盘价在 POC (Point of Control) 附近的股票（逻辑：close > poc * 0.99）。
    """
    def __init__(self, daily_context: DailyContext):
        self.context = daily_context

    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None, 
               threshold: float = 0.01) -> List[str]:
        """
        选择收盘价大于 POC * (1 - threshold) 的股票。
        
        Args:
            date: 目标日期
            candidate_codes: 候选股票代码列表
            threshold: 阈值系数，默认 0.01 (即 1 - 0.01 = 0.99)
            
        Returns:
            符合条件的股票 code 列表
        """
        # 获取目标日期的数据
        df = self.context.get_window(date, window_days=1, codes=candidate_codes)
        
        if df.empty:
            return []

        # 筛选: close > poc * (1 - threshold)
        selected = df[df['close'] > df['poc'] * (1 - threshold)]
        
        return selected['code'].tolist()

class NegativeSkewSelector:
    """
    选股器：选择偏度 (Skewness) 小于 0 的股票。
    """
    def __init__(self, daily_context: DailyContext):
        self.context = daily_context

    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None) -> List[str]:
        """
        选择指定日期偏度小于 0 的股票。
        
        Args:
            date: 目标日期
            candidate_codes: 候选股票代码列表
            
        Returns:
            符合条件的股票 code 列表
        """
        # 获取目标日期的数据
        df = self.context.get_window(date, window_days=1, codes=candidate_codes)
        
        if df.empty:
            return []

        # 筛选: skew < 0
        selected = df[df['skew'] < 0]
        
        return selected['code'].tolist()

class TopKurtosisSelector:
    """
    选股器：选择峰度 (Kurtosis) 最大的前 N 只股票。
    """
    def __init__(self, daily_context: DailyContext):
        self.context = daily_context

    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None,
               top_n: int = 5) -> List[str]:
        """
        选择指定日期峰度最大的前 N 只股票。
        
        Args:
            date: 目标日期
            candidate_codes: 候选股票代码列表
            top_n: 返回的数量
            
        Returns:
            符合条件的股票 code 列表
        """
        # 获取目标日期的数据
        df = self.context.get_window(date, window_days=1, codes=candidate_codes)
        
        if df.empty:
            return []

        # 按峰度降序排列
        df_sorted = df.sort_values('kurt', ascending=False)
        
        # 返回前 N 个
        return df_sorted.head(top_n)['code'].tolist()

class PrevDayNegativeReturnSelector:
    """
    选股器：选择前一天涨幅小于 0 的股票。
    """
    def __init__(self, daily_context: DailyContext):
        self.context = daily_context

    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None) -> List[str]:
        """
        选择以传入日期为准，前一个交易日涨幅小于 0 的股票。
        
        Args:
            date: 当前日期
            candidate_codes: 候选股票代码列表
            
        Returns:
            符合条件的股票 code 列表
        """
        # 获取当前日期和前一个交易日的数据
        # window_days=2 包含了 date 和 date 之前的一个交易日
        df = self.context.get_window(date, window_days=2, codes=candidate_codes)
        
        if df.empty:
            return []

        # 获取所有日期并排序
        all_dates = sorted(df['date'].unique())
        if len(all_dates) < 2:
            # 数据不足以判断“前一天”
            return []
        
        prev_date = all_dates[-2]
        
        # 筛选出前一天的数据
        prev_df = df[df['date'] == prev_date]
        
        # 筛选涨幅小于 0 的股票
        selected = prev_df[prev_df['pct_chg'] < 0]
        
        return selected['code'].tolist()

class PrevDayAmplitudeSelector:
    """
    选股器：选择前一天振幅不超过指定阈值（默认 3%）的股票。
    """
    def __init__(self, daily_context: DailyContext):
        self.context = daily_context

    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None,
               threshold: float = 0.03) -> List[str]:
        """
        选择前一交易日振幅 <= threshold 的股票。
        
        Args:
            date: 当前日期
            candidate_codes: 候选列表
            threshold: 振幅阈值
            
        Returns:
            符合条件的股票 code 列表
        """
        # 获取当前日期和前一个交易日的数据
        df = self.context.get_window(date, window_days=2, codes=candidate_codes)
        
        if df.empty:
            return []

        all_dates = sorted(df['date'].unique())
        if len(all_dates) < 2:
            return []
        
        prev_date = all_dates[-2]
        prev_df = df[df['date'] == prev_date]
        
        # 筛选振幅 <= threshold 的股票
        # 排除 amplitude 为 NaN 的情况
        selected = prev_df[prev_df['amplitude'] <= threshold]
        
        return selected['code'].tolist()

class AfternoonStrongSelector:
    """
    选股器：选择下午均价大于上午均价一定比例（默认 0.5%）的股票。
    """
    def __init__(self, daily_context: DailyContext):
        self.context = daily_context

    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None,
               threshold: float = 0.005) -> List[str]:
        """
        选择指定日期下午均价大于上午均价的股票。
        
        Args:
            date: 目标日期
            candidate_codes: 候选股票代码列表
            threshold: 比例阈值，默认 0.005 (0.5%)
            
        Returns:
            符合条件的股票 code 列表
        """
        # 获取目标日期的数据
        df = self.context.get_window(date, window_days=1, codes=candidate_codes)
        
        if df.empty:
            return []

        # 筛选: afternoon_mean > morning_mean * (1 + threshold)
        # 排除 morning_mean 为 0 的情况
        mask = (df['morning_mean'] > 0)
        selected = df[mask & (df['afternoon_mean'] > df['morning_mean'] * (1 + threshold))]
        
        return selected['code'].tolist()

class VReversalSelector:
    """
    选股器：选择当日呈现 V 型反转走势的股票。
    逻辑：从开盘到最低点跌幅超过 threshold_drop，从最低点到收盘回升超过 threshold_recover。
    """
    def __init__(self, daily_context: DailyContext):
        self.context = daily_context

    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None,
               threshold_drop: float = 0.02,
               threshold_recover: float = 0.02) -> List[str]:
        """
        Args:
            date: 目标日期
            candidate_codes: 候选股票代码列表
            threshold_drop: 下跌幅度阈值 (open -> min)
            threshold_recover: 回升幅度阈值 (min -> close)
            
        Returns:
            符合条件的股票 code 列表
        """
        df = self.context.get_window(date, window_days=1, codes=candidate_codes)
        if df.empty:
            return []

        # 计算跌幅和回升幅
        # 排除 open 为 0 或 min 为 0 的情况
        df = df[(df['open'] > 0) & (df['min'] > 0)].copy()
        
        df['drop_ratio'] = (df['open'] - df['min']) / df['open']
        df['recover_ratio'] = (df['close'] - df['min']) / df['min']
        
        # 触底时间过滤：最低点应在 10:00 到 14:30 之间 (假设 YYYYMMDDHHMMSSmmm 格式)
        # HHMM 在 index 8:12
        def is_valid_time(t_str):
            if not t_str or len(t_str) < 12: return False
            hhmm = t_str[8:12]
            return '1000' <= hhmm <= '1430'

        df['valid_time'] = df['min_time'].apply(is_valid_time)

        # 筛选
        mask = (df['drop_ratio'] >= threshold_drop) & \
               (df['recover_ratio'] >= threshold_recover) & \
               (df['valid_time'])
        
        selected = df[mask]
        
        return selected['code'].tolist()
