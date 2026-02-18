import pandas as pd
import clickhouse_connect
from typing import List, Optional
from stock.database.base import BaseRepository
from stock.config import settings

class ClickHouseRepository(BaseRepository):
    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=settings.CH_HOST,
            port=settings.CH_PORT,
            username=settings.CH_USER,
            password=settings.CH_PASSWORD,
            database=settings.CH_DATABASE
        )

    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        # clickhouse-connect supports params in query_df
        return self.client.query_df(sql, parameters=params)

    def insert_df(self, df: pd.DataFrame, table_name: str):
        self.client.insert_df(table_name, df)

    def execute(self, sql: str):
        self.client.command(sql)

    def get_all_dates(self, table_name: str) -> List[str]:
        res = self.query(f"SELECT DISTINCT date FROM {table_name} ORDER BY date ASC")
        return res['date'].tolist() if not res.empty else []

    def get_max_date_for_codes(self, codes: List[str], table_name: str) -> dict:
        if not codes:
            return {}
        codes_str = ", ".join([f"'{c}'" for c in codes])
        query = f"SELECT code, max(date) as max_date FROM {table_name} WHERE code IN ({codes_str}) GROUP BY code"
        res = self.query(query)
        return dict(zip(res['code'], res['max_date'])) if not res.empty else {}

    def get_min_date_for_codes(self, codes: List[str], table_name: str) -> dict:
        if not codes:
            return {}
        codes_str = ", ".join([f"'{c}'" for c in codes])
        query = f"SELECT code, min(date) as min_date FROM {table_name} WHERE code IN ({codes_str}) GROUP BY code"
        res = self.query(query)
        return dict(zip(res['code'], res['min_date'])) if not res.empty else {}

    def optimize_table(self, table_name: str):
        """Force a final merge to physically remove duplicates."""
        self.execute(f"OPTIMIZE TABLE {table_name} FINAL")

    def close(self):
        self.client.close()

    def create_mintues5_table(self):
        query = f"""
        CREATE TABLE IF NOT EXISTS {settings.TABLE_MIN5} (
            date String,
            time String,
            code String,
            open Float64,
            high Float64,
            low Float64,
            close Float64,
            volume Int64,
            amount Float64,
            adjustflag String
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (code, date, time)
        """
        self.execute(query)

    def create_daily_kline_table(self):
        # Using columns defined in DailyContext.COLUMNS
        query = f"""
        CREATE TABLE IF NOT EXISTS {settings.TABLE_DAILY} (
            date String,
            code String,
            open Float64,
            high Float64,
            low Float64,
            close Float64,
            volume Int64,
            amount Float64,
            real_price Float64,
            skew Float64,
            kurt Float64,
            poc Float64,
            morning_mean Float64,
            afternoon_mean Float64,
            min_time String,
            ksp_score Float64,
            ksp_sum_14d Float64,
            ksp_sum_7d Float64,
            ksp_sum_5d Float64,
            ksp_rank Int32,
            ksp_sum_14d_rank Int32,
            ksp_sum_7d_rank Int32,
            ksp_sum_5d_rank Int32,
            list_days Int32,
            pct_chg_skew_22d Float64,
            pct_chg_kurt_10d Float64,
            net_mf Float64,
            net_mf_5d Float64,
            net_mf_10d Float64,
            net_mf_20d Float64,
            ret_5d Float64,
            ret_10d Float64,
            ret_20d Float64,
            turn Float64
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (code, date)
        """
        self.execute(query)

    def create_concept_tables(self):
        query = f"""
        CREATE TABLE IF NOT EXISTS {settings.TABLE_CONCEPT_CONSTITUENT_THS} (
            code String,
            concept String
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (concept, code)
        """
        self.execute(query)

    def create_benchmark_table(self):
        query = f"""
        CREATE TABLE IF NOT EXISTS {settings.TABLE_BENCHMARK} (
            date String,
            code String,
            open Float64,
            high Float64,
            low Float64,
            close Float64,
            volume Int64,
            amount Float64
        ) ENGINE = ReplacingMergeTree()
        ORDER BY (code, date)
        """
        self.execute(query)
