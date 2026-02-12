import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
from selector.base_selector import BaseSelector

class RecentDaysPctChgSelector(BaseSelector):
    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None, 
               days: int = 5, 
               threshold: float = 0.05) -> List[str]:
        df = self.get_data(date, window_days=days, candidate_codes=candidate_codes)
        if df.empty: return []

        def calc_total_ret(group):
            returns = group['pct_chg'].dropna()
            if len(returns) == 0: return np.nan
            return (1 + returns).prod() - 1

        stats = df.groupby('code').apply(calc_total_ret, include_groups=False)
        return stats[stats.abs() <= threshold].index.tolist()

class MaxDailyPctChgSelector(BaseSelector):
    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None, 
               days: int = 5, 
               threshold: float = 0.05) -> List[str]:
        df = self.get_data(date, window_days=days, candidate_codes=candidate_codes)
        if df.empty: return []
        stats = df.groupby('code')['pct_chg'].apply(lambda x: x.abs().max())
        return stats[stats <= threshold].index.tolist()

class POCNearSelector(BaseSelector):
    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None, 
               threshold: float = 0.01) -> List[str]:
        df = self.get_data(date, window_days=1, candidate_codes=candidate_codes)
        if df.empty: return []
        return df[df['close'] > df['poc'] * (1 - threshold)]['code'].tolist()

class NegativeSkewSelector(BaseSelector):
    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None) -> List[str]:
        df = self.get_data(date, window_days=1, candidate_codes=candidate_codes)
        if df.empty: return []
        return df[df['skew'] < 0]['code'].tolist()

class TopKurtosisSelector(BaseSelector):
    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None,
               top_n: int = 5) -> List[str]:
        df = self.get_data(date, window_days=1, candidate_codes=candidate_codes)
        if df.empty: return []
        return df.sort_values('kurt', ascending=False).head(top_n)['code'].tolist()

class PrevDayNegativeReturnSelector(BaseSelector):
    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None) -> List[str]:
        df = self.get_data(date, window_days=2, candidate_codes=candidate_codes)
        if df.empty: return []
        all_dates = sorted(df['date'].unique())
        if len(all_dates) < 2: return []
        prev_df = df[df['date'] == all_dates[-2]]
        return prev_df[prev_df['pct_chg'] < 0]['code'].tolist()

class PrevDayAmplitudeSelector(BaseSelector):
    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None,
               threshold: float = 0.03) -> List[str]:
        df = self.get_data(date, window_days=2, candidate_codes=candidate_codes)
        if df.empty: return []
        all_dates = sorted(df['date'].unique())
        if len(all_dates) < 2: return []
        prev_df = df[df['date'] == all_dates[-2]]
        return prev_df[prev_df['amplitude'] <= threshold]['code'].tolist()

class AfternoonStrongSelector(BaseSelector):
    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None,
               threshold: float = 0.005) -> List[str]:
        df = self.get_data(date, window_days=1, candidate_codes=candidate_codes)
        if df.empty: return []
        mask = (df['morning_mean'] > 0)
        return df[mask & (df['afternoon_mean'] > df['morning_mean'] * (1 + threshold))]['code'].tolist()

class VReversalSelector(BaseSelector):
    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None,
               threshold_drop: float = 0.02,
               threshold_recover: float = 0.02) -> List[str]:
        df = self.get_data(date, window_days=1, candidate_codes=candidate_codes)
        if df.empty: return []
        df = df[(df['open'] > 0) & (df['min'] > 0)].copy()
        df['drop_ratio'] = (df['open'] - df['min']) / df['open']
        df['recover_ratio'] = (df['close'] - df['min']) / df['min']
        def is_valid_time(t_str):
            if not t_str or len(t_str) < 12: return False
            return '1000' <= t_str[8:12] <= '1430'
        df['valid_time'] = df['min_time'].apply(is_valid_time)
        mask = (df['drop_ratio'] >= threshold_drop) & (df['recover_ratio'] >= threshold_recover) & (df['valid_time'])
        return df[mask]['code'].tolist()
