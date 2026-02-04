import pandas as pd
from typing import List, Optional
from datetime import datetime

class Minutes5Context:
    """
    五分钟K线专用上下文。
    """
    def __init__(self, df: pd.DataFrame):
        self._data = df
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
    def trading_dates(self) -> List[str]:
        return self._all_dates

class MarketContext:
    """
    通用的行情上下文容器。
    策略或选股器通过它访问不同频率的数据。
    """
    def __init__(self, minutes5: Optional[Minutes5Context] = None):
        self.minutes5 = minutes5
        self.current_date: Optional[datetime] = None
        self.candidate_codes: Optional[List[str]] = None
