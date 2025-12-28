import sqlite3
import pandas as pd
from src.data.repositories.base import BaseRepository
from src.common.enums import TABLE_INDEX_STRATEGY #, TableName, Fields


class SqliteRepository(BaseRepository):
    """
    一个通用的 SQLite 仓库实现, 用于保存和加载 pandas DataFrame。
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        print(f"SqliteRepository 已初始化, 目标数据库: {db_path}")

    def __enter__(self):
        self.connection = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()

    def save(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace', save_index: bool = False):
        """
        将一个 DataFrame 保存到 SQLite 数据库的指定表中。

        :param df: 要保存的 pandas DataFrame。
        :param table_name: 要写入的数据表名称。
        :param if_exists: 如果表已存在的处理方式 ('fail', 'replace', 'append')。
        :param save_index: 是否将 DataFrame 的索引作为一列保存。
        """
        if not self.connection:
            raise ConnectionError("数据库未连接。请在 with 语句中使用该仓库。")

        print(f"正在将数据保存到数据库 {self.db_path} 的表 {table_name} 中 (保存索引: {save_index})...")
        df.to_sql(table_name, self.connection, if_exists=if_exists, index=save_index)
        print("数据保存成功。")
        
        # 保存后自动创建索引
        self._create_indexes(table_name)

    def _create_indexes(self, table_name: str):
        """ 根据预定义的策略为表创建索引。 """
        if table_name in TABLE_INDEX_STRATEGY:
            columns_to_index = TABLE_INDEX_STRATEGY[table_name]
            index_name = f"idx_{table_name}_{'_'.join(columns_to_index)}"
            
            cursor = self.connection.cursor()
            
            # 检查索引是否已存在
            cursor.execute(f"PRAGMA index_list('{table_name}')")
            existing_indexes = [row[1] for row in cursor.fetchall()]
            if index_name in existing_indexes:
                # print(f"索引 {index_name} 已存在于表 {table_name}。")
                return

            print(f"正在为表 {table_name} 在列 {columns_to_index} 上创建索引 {index_name}...")
            try:
                cols_str = ", ".join([f'"{col}"' for col in columns_to_index])
                cursor.execute(f"CREATE INDEX {index_name} ON \"{table_name}\" ({cols_str})")
                self.connection.commit()
                print("索引创建成功。")
            except sqlite3.OperationalError as e:
                # 并发场景下，索引可能刚刚被另一个连接创建
                if "already exists" in str(e):
                    print(f"索引 {index_name} 已被其他进程创建。")
                else:
                    raise e

    def get_latest_date(self, table_name: str, date_column: str) -> pd.Timestamp | None:
        """
        获取指定表中日期时间列的最新（最大）日期。

        :param table_name: 要查询的数据表名称。
        :param date_column: 日期时间列的名称。
        :return: 最新的日期 (pd.Timestamp)，如果表为空或不存在则返回 None。
        """
        if not self.connection:
            raise ConnectionError("数据库未连接。请在 with 语句中使用该仓库。")

        cursor = self.connection.cursor()
        try:
            # 检查表是否存在
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if cursor.fetchone() is None:
                print(f"表 {table_name} 不存在，将进行全量同步。")
                return None

            query = f"SELECT MAX({date_column}) FROM {table_name}"
            cursor.execute(query)
            latest_date_str = cursor.fetchone()[0]
            
            if latest_date_str:
                return pd.to_datetime(latest_date_str)
            return None
        except Exception as e:
            print(f"查询最新日期时出错: {e}")
            return None

    def load(self, table_name: str, filters: dict | None = None, order_by: str | None = None, limit: int | None = None, index_col: str | list[str] | None = None) -> pd.DataFrame:
        """
        从数据库的指定表中, 加载数据, 支持过滤和索引设置。

        :param table_name: 要查询的数据表名称。
        :param filters: (可选) 用于构建 WHERE 子句的过滤条件字典, 例如 {'symbol': 'AAPL'}。
        :param index_col: (可选) 要设置为 DataFrame 索引的列名。
        :return: 一个包含所请求数据的 pandas DataFrame。
        """
        if not self.connection:
            raise ConnectionError("数据库未连接。请在 with 语句中使用该仓库。")

        query = f"SELECT * FROM {table_name}"
        params = []

        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(f"{key} = ?")
                params.append(value)
            query += " WHERE " + " AND ".join(conditions)

        print(f"正在从表 {table_name} 加载数据, 查询语句: {query}")
        
        try:
            df = pd.read_sql(query, self.connection, params=params, index_col=index_col)
            if df.empty:
                print(f"警告: 在表 {table_name} 中未找到满足条件的数据。")
            else:
                print("数据加载成功。")
            return df
        except Exception as e:
            if "no such table" in str(e):
                raise ValueError(f"数据库 {self.db_path} 中不存在表 {table_name}。")
            # 处理 index_col 不存在的情况
            if "does not have" in str(e) and "index_col" in str(e):
                raise ValueError(f"指定的索引列 {index_col} 在表 {table_name} 中不存在。")
            raise e