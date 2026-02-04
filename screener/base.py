from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime

from stock.data.context import MarketContext

class BaseScreener(ABC):
    """
    Abstract Base Class for Stock Screeners/Alpha Models.
    """
    def __init__(self, filter_candidates: bool = True):
        self.filter_candidates = filter_candidates
        
    @abstractmethod
    def scan(self, context: MarketContext) -> pd.DataFrame:
        pass
