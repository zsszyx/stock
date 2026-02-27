import pandas as pd
from sqlalchemy import create_engine, text
from typing import List, Optional, Dict
from stock.database.base import BaseRepository
from stock.config import settings

class SQLiteRepository(BaseRepository):
    def __init__(self, db_path: str = None):
        url = db_path or settings.SQLITE_URL
        self.engine = create_engine(url, connect_args={'timeout': 30})
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL;"))
            conn.execute(text("PRAGMA synchronous=NORMAL;"))

    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        try:
            with self.engine.connect() as conn:
                if params:
                    return pd.read_sql(text(sql), conn, params=params)
                else:
                    return pd.read_sql(sql, conn)
        except Exception as e:
            print(f"SQLite Query error: {e}")
            return pd.DataFrame()

    def insert_df(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        with self.engine.begin() as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)

    def get_all_dates(self, table_name: str) -> List[str]:
        res = self.query(f"SELECT DISTINCT date FROM {table_name} ORDER BY date ASC")
        return res['date'].tolist() if not res.empty else []

    def get_max_date_for_codes(self, codes: List[str], table_name: str) -> Dict[str, str]:
        if not codes: return {}
        placeholders = ', '.join(['?'] * len(codes))
        query = f"SELECT code, max(date) as max_date FROM {table_name} WHERE code IN ({placeholders}) GROUP BY code"
        res = self.query(query, params=codes)
        return dict(zip(res['code'], res['max_date'])) if not res.empty else {}

    def get_min_date_for_codes(self, codes: List[str], table_name: str) -> Dict[str, str]:
        if not codes: return {}
        placeholders = ', '.join(['?'] * len(codes))
        query = f"SELECT code, min(date) as min_date FROM {table_name} WHERE code IN ({placeholders}) GROUP BY code"
        res = self.query(query, params=codes)
        return dict(zip(res['code'], res['min_date'])) if not res.empty else {}

    def close(self):
        if self.engine:
            self.engine.dispose()

    def execute(self, sql: str):
        with self.engine.begin() as conn:
            conn.execute(text(sql))

    def create_mintues5_table(self):
        # Implementation for SQLite if needed
        pass

    def create_daily_kline_table(self):
        # Implementation for SQLite if needed
        pass

    def create_concept_tables(self):
        # Already exists in stock.db
        pass

    def create_benchmark_table(self):
        pass

    def optimize_table(self, table_name: str):
        # SQLite uses VACUUM for optimization, but we can skip it for simple tasks
        pass
