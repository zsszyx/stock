from abc import ABC, abstractmethod
from typing import Iterator, List, Dict, Optional
import pandas as pd
from datetime import datetime
from .models import Bar

class DataFeed(ABC):
    """Abstract Base Class for Data Feeds"""
    
    @abstractmethod
    def get_full_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取全市场的全量数据（用于 Context）"""
        pass

    @abstractmethod
    def set_universe(self, codes: List[str], start_date: str, end_date: str) -> None:
        """设定回测的具体股票池和范围"""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Bar]]:
        pass

class SqliteDataFeed(DataFeed):
    """DataFeed implementation for the project's SQLite database"""
    
    def __init__(self, sql_op, table_name: str = "mintues5"):
        self.sql_op = sql_op
        self.table_name = table_name
        self._full_cache: pd.DataFrame = pd.DataFrame() # 存放全量数据
        self._backtest_data: pd.DataFrame = pd.DataFrame() # 存放回测步进数据

    def get_full_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        print(f"Loading full market data for context: {start_date} to {end_date}")
        query = f"""
            SELECT code, date, time, open, high, low, close, volume, amount 
            FROM {self.table_name}
            WHERE date >= '{start_date}' AND date <= '{end_date}'
        """
        df = self.sql_op.query(query)
        if df is not None and not df.empty:
            df['datetime'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M%S%f')
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            self._full_cache = df
        return self._full_cache

    def set_universe(self, codes: List[str], start_date: str, end_date: str) -> None:
        """
        根据确定的 Universe 准备步进迭代的数据。
        """
        if not self._full_cache.empty:
            # 性能优化：如果已经有全量缓存，直接在内存过滤
            print(f"Filtering universe from memory ({len(codes)} stocks)...")
            self._backtest_data = self._full_cache[self._full_cache['code'].isin(codes)].copy()
        else:
            # 降级：去数据库取
            print(f"Loading universe from DB ({len(codes)} stocks)...")
            codes_str = "'" + "','".join(codes) + "'"
            query = f"""
                SELECT code, date, time, open, high, low, close, volume, amount 
                FROM {self.table_name}
                WHERE code IN ({codes_str})
                AND date >= '{start_date}' AND date <= '{end_date}'
            """
            df = self.sql_op.query(query)
            if df is not None and not df.empty:
                df['datetime'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M%S%f')
                for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                self._backtest_data = df
        
        if not self._backtest_data.empty:
            self._backtest_data = self._backtest_data.sort_values('datetime')

    def __iter__(self) -> Iterator[Dict[str, Bar]]:
        if self._backtest_data.empty:
            return

        grouped = self._backtest_data.groupby('datetime')
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
                    amount=row['amount'],
                    code=row['code']
                )
            yield bars