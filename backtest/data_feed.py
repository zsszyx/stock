from abc import ABC, abstractmethod
from typing import Iterator, List, Dict
import pandas as pd
from datetime import datetime
from .models import Bar

class DataFeed(ABC):
    """Abstract Base Class for Data Feeds"""
    
    @abstractmethod
    def load_data(self, codes: List[str], start_date: str, end_date: str) -> None:
        """Pre-fetch or prepare data stream"""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Bar]]:
        """
        Yields bars for each time step.
        Returns a dict mapping {code: Bar} for that timestamp.
        """
        pass

class SqliteDataFeed(DataFeed):
    """DataFeed implementation for the project's SQLite database"""
    
    def __init__(self, sql_op, table_name: str = "mintues5"):
        self.sql_op = sql_op
        self.table_name = table_name
        self._data_cache: pd.DataFrame = pd.DataFrame()
        self._codes = []

    def load_data(self, codes: List[str], start_date: str, end_date: str) -> None:
        self._codes = codes
        # Construct query carefully to handle multiple codes
        codes_str = "'" + "','".join(codes) + "'"
        query = f"""
            SELECT code, time, open, high, low, close, volume 
            FROM {self.table_name}
            WHERE code IN ({codes_str})
            AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY time ASC
        """
        # Load all into memory for speed (Pandas is fast), then iterate
        # For massive datasets, we would implement chunking here.
        self._data_cache = self.sql_op.query(query)
        
        # Pre-process time
        if not self._data_cache.empty:
            # Format: 20251201093500000 -> Datetime
            self._data_cache['datetime'] = pd.to_datetime(self._data_cache['time'], format='%Y%m%d%H%M%S%f')
            
            # Ensure numeric types
            cols = ['open', 'high', 'low', 'close', 'volume']
            for col in cols:
                self._data_cache[col] = pd.to_numeric(self._data_cache[col])

    def __iter__(self) -> Iterator[Dict[str, Bar]]:
        if self._data_cache.empty:
            return

        # Group by datetime to simulate market time steps
        grouped = self._data_cache.groupby('datetime')
        
        for timestamp, group in grouped:
            bars = {}
            for _, row in group.iterrows():
                bars[row['code']] = Bar(
                    time=timestamp,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    code=row['code']
                )
            yield bars
