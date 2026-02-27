import io
import logging
import os
import subprocess
import time
from typing import List, Optional

import pandas as pd
from clickhouse_driver import Client

from stock.config import settings
from stock.database.base import BaseRepository

logger = logging.getLogger(__name__)

class ClickHouseRepository(BaseRepository):
    """
    ClickHouse 仓库类 - CLI Bridge 模式
    通过直接调用 clickhouse client 命令行工具来绕过 macOS 权限隔离和网络库不稳定的问题。
    """
    def __init__(self, max_retries: int = 3, retry_interval: int = 5):
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        # 验证 clickhouse client 是否可用
        self._test_connection()

    def _test_connection(self):
        try:
            res = self._run_cli("SELECT 1")
            if res.strip() == "1":
                logger.info("CLI Bridge Connected: clickhouse-client is functional.")
        except Exception as e:
            logger.error(f"CLI Bridge Failure: {str(e)}")

    def _run_cli(self, sql: str, format: str = "CSVWithNames") -> str:
        """执行 clickhouse-client 命令并返回输出字符串"""
        cmd = [
            "clickhouse", "client",
            "--host", "127.0.0.1",
            "--port", "9000",
            "--query", sql,
            "--format", format
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = f"ClickHouse CLI Error: {result.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) # Fail Fast: 立即抛出异常，阻止后续逻辑运行
        return result.stdout

    def _log_error(self, message: str):
        # 兼容旧代码，将日志记录转发给 logger
        logger.error(message)

    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        # 将参数手动替换到 SQL 中 (CLI 不支持参数化)
        if params:
            for k, v in params.items():
                escaped = v.replace("'", "''") if isinstance(v, str) else v
                val = f"'{escaped}'" if isinstance(v, str) else str(v)
                sql = sql.replace(f"{{{k}}}", val)
                sql = sql.replace(f":{k}", val)

        last_exception = None
        for i in range(self.max_retries):
            try:
                # 使用 CSVWithNames 格式方便 pandas 解析
                output = self._run_cli(sql, format="CSVWithNames")
                if not output.strip():
                    return pd.DataFrame()
                return pd.read_csv(io.StringIO(output))
            except Exception as e:
                last_exception = e
                self._log_error(f"CLI Query attempt {i+1} failed: {str(e)}")
                time.sleep(self.retry_interval)
        
        self._log_error(f"CLI Query failed: {str(last_exception)}")
        return pd.DataFrame()

    def insert_df(self, df: pd.DataFrame, table_name: str):
        if df.empty: return
        # 使用 CSV 格式进行高性能插入
        csv_data = df.to_csv(index=False, header=False)
        cmd = [
            "clickhouse", "client", "--host", "127.0.0.1", "--port", "9000",
            "--query", f"INSERT INTO {table_name} FORMAT CSV"
        ]
        try:
            subprocess.run(cmd, input=csv_data, text=True, check=True)
        except Exception as e:
            self._log_error(f"CLI Insert failed: {str(e)}")

    def execute(self, sql: str):
        try:
            self._run_cli(sql, format="Null")
        except Exception as e:
            self._log_error(f"CLI Execute failed: {str(e)}")

    def get_all_dates(self, table_name: str) -> List[str]:
        res = self.query(f"SELECT DISTINCT date FROM {table_name} ORDER BY date ASC")
        return res['date'].tolist() if not res.empty else []

    def get_max_date_for_codes(self, codes: List[str], table_name: str) -> dict:
        if not codes: return {}
        codes_str = ", ".join([f"'{c}'" for c in codes])
        query = f"SELECT code, max(date) as max_date FROM {table_name} WHERE code IN ({codes_str}) GROUP BY code"
        res = self.query(query)
        return dict(zip(res['code'], res['max_date'])) if not res.empty else {}

    def get_min_date_for_codes(self, codes: List[str], table_name: str) -> dict:
        if not codes: return {}
        codes_str = ", ".join([f"'{c}'" for c in codes])
        query = f"SELECT code, min(date) as min_date FROM {table_name} WHERE code IN ({codes_str}) GROUP BY code"
        res = self.query(query)
        return dict(zip(res['code'], res['min_date'])) if not res.empty else {}

    def optimize_table(self, table_name: str):
        self.execute(f"OPTIMIZE TABLE {table_name} FINAL")

    def close(self):
        # CLI 模式不需要显式关闭连接
        pass

    def create_mintues5_table(self):
        query = f"""
        CREATE TABLE IF NOT EXISTS {settings.TABLE_MIN5} (
            date String, time String, code String, open Float64, high Float64, low Float64, close Float64, volume Int64, amount Float64, adjustflag String
        ) ENGINE = ReplacingMergeTree() ORDER BY (code, date, time)
        """
        self.execute(query)

    def create_daily_kline_table(self):
        query = f"""
        CREATE TABLE IF NOT EXISTS {settings.TABLE_DAILY} (
            date String, code String, open Float64, high Float64, low Float64, close Float64, volume Int64, amount Float64, real_price Float64, skew Float64, kurt Float64, poc Float64, morning_mean Float64, afternoon_mean Float64, min_time String, ksp_score Float64, ksp_sum_14d Float64, ksp_sum_10d Float64, ksp_sum_7d Float64, ksp_sum_5d Float64, ksp_rank Int32, ksp_sum_14d_rank Int32, ksp_sum_10d_rank Int32, ksp_sum_7d_rank Int32, ksp_sum_5d_rank Int32, list_days Int32, pct_chg_skew_22d Float64, pct_chg_kurt_10d Float64, net_mf Float64, net_mf_5d Float64, net_mf_10d Float64, net_mf_20d Float64, ret_5d Float64, ret_10d Float64, ret_20d Float64, turn Float64
        ) ENGINE = ReplacingMergeTree() ORDER BY (code, date)
        """
        self.execute(query)

    def create_concept_tables(self):
        query = f"""
        CREATE TABLE IF NOT EXISTS {settings.TABLE_CONCEPT_CONSTITUENT_THS} (
            code String, concept String
        ) ENGINE = ReplacingMergeTree() ORDER BY (concept, code)
        """
        self.execute(query)

    def create_benchmark_table(self):
        query = f"""
        CREATE TABLE IF NOT EXISTS {settings.TABLE_BENCHMARK} (
            date String, code String, open Float64, high Float64, low Float64, close Float64, volume Int64, amount Float64
        ) ENGINE = ReplacingMergeTree() ORDER BY (code, date)
        """
        self.execute(query)
