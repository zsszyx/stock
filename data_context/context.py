import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
from stock.factors.distribution_analyzer import DistributionAnalyzer

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
        'ksp_score', 'ksp_sum_14d', 'ksp_sum_7d', 'ksp_sum_5d',
        'ksp_rank', 'ksp_sum_14d_rank', 'ksp_sum_7d_rank', 'ksp_sum_5d_rank',
        'list_days', 'pct_chg_skew_22d', 'pct_chg_kurt_10d', 'turn'
    ]

    def __init__(self, min5_context: Optional[Minutes5Context] = None, daily_df: Optional[pd.DataFrame] = None):
        if daily_df is not None: self._daily_df = daily_df.copy()
        elif min5_context is not None: self._daily_df = self.from_min5(min5_context.data)
        else: raise ValueError("Either min5_context or daily_df must be provided")
        if not self._daily_df.empty: self._daily_df = self._add_derived_factors(self._daily_df)
        self._all_dates = sorted(self._daily_df['date'].unique())
        self._date_to_idx = {d: i for i, d in enumerate(self._all_dates)}

    @staticmethod
    def _add_derived_factors(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['code', 'date'])
        # 1. 基础涨跌幅 (使用真实收盘价)
        df['prev_close'] = df.groupby('code')['close'].shift(1)
        df['pct_chg'] = (df['close'] - df['prev_close']) / df['prev_close']
        df['amplitude'] = (df['high'] - df['low']) / df['prev_close']
        
        # 2. 核心 KSP 分数逻辑 (严格按照您的 3% 惩罚标准，且使用 5分钟均价派生出的统计指标)
        base_magnitude = df['kurt'].abs() * df['skew'].abs()
        magnitude = np.where(df['pct_chg'].abs() > 0.03, 
                             base_magnitude * df['pct_chg'].abs() * 100, 
                             base_magnitude)
        # 风险项：偏度>0, 峰度<0, 涨幅过大, 或 POC 低于前收盘
        is_bad = (df['kurt'] < 0) | (df['skew'] > 0) | (df['pct_chg'].abs() > 0.03) | (df['poc'] < df['prev_close'] * 0.98)
        df['ksp_score'] = np.where(is_bad, -magnitude, magnitude)
        
        # 3. 滚动因子 (不进行任何填充，缺失即为 NaN)
        for p in [5, 7, 14]:
            df[f'ksp_sum_{p}d'] = df.groupby('code')['ksp_score'].transform(lambda x: x.rolling(window=p, min_periods=1).sum())
        
        df['ksp_rank'] = df.groupby('date')['ksp_score'].rank(ascending=False, method='min')
        for p in [5, 7, 14]:
            df[f'ksp_sum_{p}d_rank'] = df.groupby('date')[f'ksp_sum_{p}d'].rank(ascending=False, method='min')
        
        # 4. 上市时长 (识别数据库起始日)
        global_min_date = df['date'].min()
        df['first_date'] = df.groupby('code')['date'].transform('min')
        df['list_days'] = (pd.to_datetime(df['date']) - pd.to_datetime(df['first_date'])).dt.days
        df.loc[df['first_date'] == global_min_date, 'list_days'] += 1000

        # 5. 统计指标
        df['pct_chg_skew_22d'] = df.groupby('code')['pct_chg'].transform(lambda x: x.rolling(window=22, min_periods=10).skew())
        df['pct_chg_kurt_10d'] = df.groupby('code')['pct_chg'].transform(lambda x: x.rolling(window=10, min_periods=5).kurt())
        return df

    @staticmethod
    def from_min5(df_min5: pd.DataFrame) -> pd.DataFrame:
        if df_min5.empty: return pd.DataFrame(columns=DailyContext.COLUMNS)
        df = df_min5.copy()
        
        # --- 核心：使用 5 分钟成交均价进行统计分析 ---
        df['real_price'] = np.divide(df['amount'], df['volume'], out=np.zeros_like(df['amount']), where=df['volume']!=0)
        
        # 聚合基础日线
        agg_res = df.sort_values(['code', 'date', 'time']).groupby(['code', 'date']).agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            volume=('volume', 'sum'),
            amount=('amount', 'sum'),
            net_mf=('amount', lambda x: (x * np.where(df.loc[x.index, 'close'] > df.loc[x.index, 'open'], 1, -1)).sum())
        )
        
        # 记录每日均价
        agg_res['real_price'] = agg_res['amount'] / agg_res['volume']
        
        def calc_complex_stats(group):
            # 严格使用 5 分钟均价 (real_price) 传入 Analyzer
            prices = np.divide(group['amount'].values, group['volume'].values, out=np.zeros_like(group['amount'].values), where=group['volume'].values!=0)
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
        
        for col in DailyContext.COLUMNS:
            if col not in daily.columns: daily[col] = 0.0 if 'time' not in col else ""
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
