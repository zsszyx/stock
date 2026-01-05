import pandas as pd
from sqlalchemy import create_engine, text
from . import sql_config

class SqlOp:
    def __init__(self, db_path=sql_config.db_path):
        self.engine = create_engine(db_path)

    def upsert_df_to_db(self, df: pd.DataFrame, table_name: str, index: bool = False):
        """
        将DataFrame upsert到数据库。如果表不存在，则创建表。
        :param df: 要保存的DataFrame
        :param table_name: 数据库中的表名
        :param index: bool, default False. Write DataFrame index as a column.
        """
        if df.empty:
            return

        temp_table_name = f"temp_{table_name}"
        try:
            with self.engine.connect() as conn:
                # 检查目标表是否存在
                if not self.engine.dialect.has_table(conn, table_name):
                    # 如果表不存在，直接将DataFrame写入新表
                    df.to_sql(table_name, self.engine, if_exists='fail', index=index)
                    return

            # 如果表已存在，则执行upsert操作
            # 将DataFrame写入临时表
            df.to_sql(temp_table_name, self.engine, if_exists='replace', index=index)

            with self.engine.begin() as conn:
                # 从临时表获取列名
                temp_df_cols = pd.read_sql(f'SELECT * FROM "{temp_table_name}" LIMIT 0', conn).columns.tolist()
                cols_str = ', '.join(f'"{col}"' for col in temp_df_cols)

                # 构建INSERT OR REPLACE语句
                insert_sql = f"""
                    INSERT OR REPLACE INTO "{table_name}" ({cols_str})
                    SELECT {cols_str} FROM "{temp_table_name}"
                """
                conn.execute(text(insert_sql))
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_max_date_for_codes(self, codes: list, table_name: str) -> dict:
        """
        查询多个股票代码对应的最大日期
        :param codes: 股票代码列表
        :param table_name: 表名
        :return: 一个字典，键是股票代码，值是对应的最大日期
        """
        if not codes:
            return {}

        try:
            # 构建参数化的查询
            placeholders = ', '.join([f':code_{i}' for i in range(len(codes))])
            query_str = f"""
                SELECT code, MAX(date) as max_date
                FROM {table_name}
                WHERE code IN ({placeholders})
                GROUP BY code
            """
            params = {f'code_{i}': code for i, code in enumerate(codes)}
            
            with self.engine.connect() as connection:
                result = connection.execute(text(query_str), params)
                max_dates = {row[0]: row[1] for row in result}
            
            return max_dates
        except Exception as e:
            print(f"An error occurred during query: {e}")
            return {code: None for code in codes}

    def query(self, query_str: str) -> pd.DataFrame:
        """
        执行一个通用的SQL查询
        :param query_str: 要执行的SQL查询语句
        :return: 包含查询结果的DataFrame
        """
        try:
            return pd.read_sql(query_str, self.engine)
        except Exception as e:
            print(f"An error occurred during query: {e}")
            return None

    def read_all_k_data(self, table_name: str) -> pd.DataFrame:
        """
        读取所有K线数据
        :param table_name: 表名
        :return: 包含所有K线数据的DataFrame
        """
        return self.query(f"SELECT * FROM {table_name}")

    def read_k_data_by_date_range(self, table_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        读取指定日期范围内的K线数据
        :param table_name: 表名
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: 包含指定日期范围内K线数据的DataFrame
        """
        query_str = f"SELECT * FROM {table_name} WHERE date >= '{start_date}' AND date <= '{end_date}'"
        return self.query(query_str)

    def close(self):
        """
        关闭数据库引擎，释放所有连接。
        """
        if self.engine:
            self.engine.dispose()

if __name__ == '__main__':
    # 示例用法
    # 创建一个SqlOp实例
    sql_op = SqlOp()

    # 创建一个示例DataFrame
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df_to_save = pd.DataFrame(data)

    # 将DataFrame保存到数据库
    table_name = sql_config.mintues5_table_name
    sql_op.save_df_to_db(df_to_save, table_name)

    # 从数据库查询数据
    query_result_df = sql_op.query(f"SELECT * FROM {table_name}")
    print("Query result:")
    print(query_result_df)