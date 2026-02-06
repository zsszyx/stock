import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
from factors.distribution_analyzer import DistributionAnalyzer

class Minutes5Context:
    """
    五分钟K线专用上下文。
    """
    def __init__(self, df: pd.DataFrame):
        self._data = df.copy()
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col in self._data.columns:
                self._data[col] = pd.to_numeric(self._data[col], errors='coerce')
        
        self._all_dates = sorted(self._data['date'].unique())
        self._date_to_idx = {d: i for i, d in enumerate(self._all_dates)}
        
    def get_window(self, date: datetime, window_days: int = 1, codes: Optional[List[str]] = None) -> pd.DataFrame:
        date_str = date.strftime('%Y-%m-%d')
        if date_str not in self._date_to_idx:
            return pd.DataFrame()
            
        current_idx = self._date_to_idx[date_str]
        start_idx = max(0, current_idx - window_days + 1)
        target_dates = self._all_dates[start_idx : current_idx + 1]
        
        mask = self._data['date'].isin(target_dates)
        if codes is not None:
            mask &= self._data['code'].isin(codes)
        return self._data[mask]

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def trading_dates(self) -> List[str]:
        return self._all_dates

class DailyContext:
    """
    日线上下文，由五分钟数据聚合而成。
    """
    def __init__(self, min5_context: Minutes5Context):
        self._min5_data = min5_context.data
        self._daily_df = self._aggregate_to_daily()
        self._all_dates = sorted(self._daily_df['date'].unique())
        self._date_to_idx = {d: i for i, d in enumerate(self._all_dates)}

    def _aggregate_to_daily(self) -> pd.DataFrame:
        def calc_stats(group):
            analyzer = DistributionAnalyzer.from_amount_volume(
                amounts=group['amount'].values,
                volumes=group['volume'].values,
                times=group['time'].values
            )
            
            # Last close price of the day
            sorted_group = group.sort_values('time')
            last_close = sorted_group['close'].iloc[-1]
            
            return pd.Series({
                'skew': analyzer.skewness,
                'kurt': analyzer.kurtosis,
                'poc': analyzer.poc,
                'morning_mean': analyzer.morning_mean_price,
                'afternoon_mean': analyzer.afternoon_mean_price,
                'open': analyzer.open_price,
                'high': analyzer.max_price,
                'min': analyzer.min_price,
                'min_time': analyzer.min_time,
                'close': last_close
            })

        daily = self._min5_data.groupby(['code', 'date']).apply(calc_stats, include_groups=False).reset_index()
        
        # Calculate pct_chg: (close - prev_close) / prev_close
        daily = daily.sort_values(['code', 'date'])
        daily['prev_close'] = daily.groupby('code')['close'].shift(1)
        daily['pct_chg'] = (daily['close'] - daily['prev_close']) / daily['prev_close']
        
        # Calculate amplitude: (high - low) / prev_close
        # low is stored in 'min' column from previous implementation
        daily['amplitude'] = (daily['high'] - daily['min']) / daily['prev_close']
        
        return daily

    def get_window(self, date: datetime, window_days: int = 1, codes: Optional[List[str]] = None) -> pd.DataFrame:
        date_str = date.strftime('%Y-%m-%d')
        if date_str not in self._date_to_idx:
            return pd.DataFrame()
            
        current_idx = self._date_to_idx[date_str]
        start_idx = max(0, current_idx - window_days + 1)
        target_dates = self._all_dates[start_idx : current_idx + 1]
        
        mask = self._daily_df['date'].isin(target_dates)
        if codes is not None:
            mask &= self._daily_df['code'].isin(codes)
        return self._daily_df[mask]

    @property
    def data(self) -> pd.DataFrame:
        return self._daily_df

    @property
    def trading_dates(self) -> List[str]:
        return self._all_dates
