import pandas as pd
from typing import List, Optional
from datetime import datetime
from data_context.context import DailyContext

class BaseSelector:
    """
    Base class for all selectors to reduce boilerplate.
    """
    def __init__(self, daily_context: DailyContext):
        self.context = daily_context

    def get_data(self, date: datetime, window_days: int = 1, candidate_codes: Optional[List[str]] = None) -> pd.DataFrame:
        df = self.context.get_window(date, window_days=window_days, codes=candidate_codes)
        return df

    def select(self, date: datetime, candidate_codes: Optional[List[str]] = None, **kwargs) -> List[str]:
        raise NotImplementedError("Subclasses must implement select()")
