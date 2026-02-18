from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional

class BaseRepository(ABC):
    @abstractmethod
    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def insert_df(self, df: pd.DataFrame, table_name: str):
        pass

    @abstractmethod
    def get_all_dates(self, table_name: str) -> List[str]:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def create_mintues5_table(self):
        pass

    @abstractmethod
    def create_daily_kline_table(self):
        pass

    @abstractmethod
    def create_concept_tables(self):
        pass

    @abstractmethod
    def create_benchmark_table(self):
        pass

    @abstractmethod
    def optimize_table(self, table_name: str):
        pass

    @abstractmethod
    def execute(self, sql: str):
        pass
