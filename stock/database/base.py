from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional, Dict

class BaseRepository(ABC):
    @abstractmethod
    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def insert_df(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        pass

    @abstractmethod
    def execute(self, sql: str):
        pass

    @abstractmethod
    def get_all_dates(self, table_name: str) -> List[str]:
        pass

    @abstractmethod
    def get_max_date_for_codes(self, codes: List[str], table_name: str) -> Dict[str, str]:
        pass

    @abstractmethod
    def get_min_date_for_codes(self, codes: List[str], table_name: str) -> Dict[str, str]:
        pass

    @abstractmethod
    def optimize_table(self, table_name: str):
        pass

    @abstractmethod
    def close(self):
        pass
