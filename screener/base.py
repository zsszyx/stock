from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime

from stock.data.context import MarketContext

class BaseScreener(ABC):
    """
    Abstract Base Class for Stock Screeners/Alpha Models.
    """
        
    @abstractmethod
    def scan(self, context: MarketContext) -> pd.DataFrame:
        pass

