import sqlite3
import pandas as pd
from .base import BaseConnector

class SqliteConnector(BaseConnector):
    """
    一个通用的 SQLite 数据库连接器。
    """

    def __init__(self, db_path: str):
        """
        初始化连接器并指定数据库文件路径。

        :param db_path: SQLite 数据库文件的路径。
        """
        self.db_path = db_path
        self.connection = None

    def __enter__(self):
        """支持 with 语句, 在进入时建立数据库连接。"""
        self.connection = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """在退出 with 语句时关闭数据库连接。"""
        if self.connection:
            self.connection.close()

    def fetch(self, table_name: str, symbol: str | None = None, **kwargs) -> pd.DataFrame:
        """
        从数据库的指定表中, 获取数据。
        如果提供了 symbol, 则获取指定股票代码的数据。
        如果没有提供, 则获取表中的所有数据。

        :param table_name: 要查询的数据表名称。
        :param symbol: (可选) 要获取的股票代码。如果为 None, 则获取所有股票数据。
        :param kwargs: 预留的其他参数。
        :return: 一个包含原始数据的 pandas DataFrame。
        """
        if not self.connection:
            raise ConnectionError("数据库未连接。请在 with 语句中使用该连接器。")

        query = f"SELECT * FROM {table_name}"
        
        if symbol:
            print(f"正在从数据库 {self.db_path} 的表 {table_name} 中查询代码为 {symbol} 的股票数据...")
            query += f" WHERE symbol = '{symbol}'"
        else:
            print(f"正在从数据库 {self.db_path} 的表 {table_name} 中查询所有股票数据...")

        df = pd.read_sql(query, self.connection)
        
        if df.empty:
            if symbol:
                raise ValueError(f"在表 {table_name} 中未找到代码为 {symbol} 的数据。")
            else:
                raise ValueError(f"在表 {table_name} 中未找到任何数据。")
            
        return df