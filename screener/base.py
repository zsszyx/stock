from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime

class BaseScreener(ABC):
    """
    Abstract Base Class for Stock Screeners/Alpha Models.
    Responsibility: Select a subset of stocks from a universe based on specific logic.
    """
    
    @abstractmethod
    def scan(self, date: datetime) -> pd.DataFrame:
        """
        Scans the market for a specific date.
        
        Args:
            date: The date to perform the scan (usually looks at data up to this point).
            
        Returns:
            pd.DataFrame: A DataFrame containing at least ['code'] and relevant factor columns.
                          Should be sorted by priority/rank.
        """
        pass
