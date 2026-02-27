import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime

from stock.factors.distribution_analyzer import DistributionAnalyzer
from stock.factors.ksp_factors import KSPFactorEngine
from stock.utils.data_utils import DataUtils

class Minutes5Context:
    def __init__(self, df: pd.DataFrame):
        self._data = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col in self._data.columns: self._data[col] = pd.to_numeric(self._data[col], errors='coerce')
        self._all_dates = sorted(self._data['date'].unique())
        self._date_to_idx = {d: i for i, d in enumerate(self._all_dates)}
    def get_window(self, date: datetime, window_days: int = 1, codes: Optional[List[str]] = None) -> pd.DataFrame:
        date_str = date.strftime('%Y-%m-%d')
        if date_str not in self._date_to_idx: return pd.DataFrame()
        current_idx = self._date_to_idx[date_str]
        start_idx = max(0, current_idx - window_days + 1)
        target_dates = self._all_dates[start_idx : current_idx + 1]
        mask = self._data['date'].isin(target_dates)
        if codes is not None: mask &= self._data['code'].isin(codes)
        return self._data[mask]
    @property
    def data(self) -> pd.DataFrame: return self._data

class DailyContext:
    COLUMNS = [
        'date', 'code', 'open', 'high', 'low', 'close', 
        'volume', 'amount', 'real_price', 'skew', 'kurt', 
        'poc', 'morning_mean', 'afternoon_mean', 'min_time',
        'ksp_score', 'ksp_sum_14d', 'ksp_sum_10d', 'ksp_sum_7d', 'ksp_sum_5d',
        'ksp_rank', 'ksp_sum_14d_rank', 'ksp_sum_10d_rank', 'ksp_sum_7d_rank', 'ksp_sum_5d_rank',
        'list_days', 'pct_chg_skew_22d', 'pct_chg_kurt_10d', 
        'net_mf', 'net_mf_5d', 'net_mf_10d', 'net_mf_20d',
        'ret_5d', 'ret_10d', 'ret_20d', 'turn'
    ]

    def __init__(self, min5_context: Optional[Minutes5Context] = None, daily_df: Optional[pd.DataFrame] = None):
        if daily_df is not None: self._daily_df = daily_df.copy()
        elif min5_context is not None: self._daily_df = self.from_min5(min5_context.data)
        else: raise ValueError("Either min5_context or daily_df must be provided")
        
        # Fail Fast: 必须保证核心列存在，否则立即报错
        required = ['date', 'code', 'close', 'high', 'low']
        missing = [c for c in required if c not in self._daily_df.columns]
        if missing:
            raise ValueError(f"DailyContext data integrity check failed. Missing columns: {missing}")

        if not self._daily_df.empty: 
            self._daily_df = self._add_derived_factors(self._daily_df)
            self._daily_df = self._daily_df.sort_values(['date', 'code']).reset_index(drop=True)
            
        self._all_dates = sorted(self._daily_df['date'].unique())
        self._date_to_idx = {d: i for i, d in enumerate(self._all_dates)}
        
        # Pre-calculate date boundaries for O(1) slicing
        self._date_offsets = {}
        if not self._daily_df.empty:
            groups = self._daily_df.groupby('date').groups
            for d, idx_list in groups.items():
                self._date_offsets[d] = (idx_list[0], idx_list[-1] + 1)

    @staticmethod
    def _add_derived_factors(df: pd.DataFrame) -> pd.DataFrame:
        
        # 1. 价格指标
        df = KSPFactorEngine.add_price_metrics(df)
        
        # 2. 核心 KSP 分数
        df = KSPFactorEngine.calculate_ksp_scores(df)
        
        # 3. 滚动因子与排名
        df = KSPFactorEngine.add_rolling_factors(df)
        
        # 4. 上市时长处理
        global_min_date = df['date'].min()
        df['first_date'] = df.groupby('code')['date'].transform('min')
        df['list_days'] = (pd.to_datetime(df['date']) - pd.to_datetime(df['first_date'])).dt.days
        df.loc[df['first_date'] == global_min_date, 'list_days'] += 1000
        
        return df

    @staticmethod
    def from_min5(df_min5: pd.DataFrame) -> pd.DataFrame:
        if df_min5.empty: return pd.DataFrame(columns=DailyContext.COLUMNS)
        
        # 使用工具类统一清洗
        df = DataUtils.clean_numeric_df(df_min5.copy(), ['open', 'high', 'low', 'close', 'volume', 'amount'])
        
        # --- 核心：使用 5 分钟成交均价进行统计分析 ---
        df['real_price'] = np.divide(df['amount'], df['volume'], out=np.zeros_like(df['amount'], dtype=float), where=df['volume']!=0)
        
        # Pre-calculate money flow for net_mf to avoid slow lambda indexing
        df['bar_sign'] = np.where(df['close'] > df['open'], 1, -1)
        df['mf'] = df['amount'] * df['bar_sign']
        
        # 聚合基础日线
        agg_res = df.sort_values(['code', 'date', 'time']).groupby(['code', 'date']).agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            volume=('volume', 'sum'),
            amount=('amount', 'sum'),
            net_mf=('mf', 'sum')
        )
        
        # 记录每日均价
        agg_res['real_price'] = agg_res['amount'] / agg_res['volume']
        
        def calc_complex_stats(group):
            # 严格使用 5 分钟均价 (real_price) 传入 Analyzer
            prices = np.divide(group['amount'].values, group['volume'].values, out=np.zeros_like(group['amount'].values, dtype=float), where=group['volume'].values!=0)
            analyzer = DistributionAnalyzer(prices=prices, volumes=group['volume'].values, amounts=group['amount'].values, times=group['time'].values)
            return pd.Series({
                'skew': analyzer.skewness, 
                'kurt': analyzer.kurtosis, 
                'poc': analyzer.poc, 
                'morning_mean': analyzer.morning_mean,
                'afternoon_mean': analyzer.afternoon_mean,
                'min_time': analyzer.min_time
            })
        
        complex_stats = df.groupby(['code', 'date']).apply(calc_complex_stats, include_groups=False)
        daily = agg_res.join(complex_stats).reset_index()
        
        # 确保所有列都存在且类型正确
        int_cols = ['volume', 'ksp_rank', 'ksp_sum_14d_rank', 'ksp_sum_10d_rank', 'ksp_sum_7d_rank', 'ksp_sum_5d_rank', 'list_days']
        for col in DailyContext.COLUMNS:
            if col not in daily.columns:
                if col in int_cols:
                    if 'rank' in col:
                        daily[col] = 5000
                    else:
                        daily[col] = 0
                elif 'time' in col or col in ['date', 'code']:
                    daily[col] = ""
                else:
                    daily[col] = 0.0
        
        return daily[DailyContext.COLUMNS]

    def get_window(self, date: datetime, window_days: int = 1, codes: Optional[List[str]] = None) -> pd.DataFrame:
        date_str = date.strftime('%Y-%m-%d')
        if date_str not in self._date_to_idx: return pd.DataFrame()
        current_idx = self._date_to_idx[date_str]
        start_idx = max(0, current_idx - window_days + 1)
        target_dates = self._all_dates[start_idx : current_idx + 1]
        mask = self._daily_df['date'].isin(target_dates)
        if codes is not None: mask &= self._daily_df['code'].isin(codes)
        return self._daily_df[mask]
    @property
    def data(self) -> pd.DataFrame: return self._daily_df
    @property
    def trading_dates(self) -> List[str]: return self._all_dates
