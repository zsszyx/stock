import pandas as pd
from src.data.repositories.base import BaseRepository
from src.common.enums import TableName, Fields

class TradingDataQuery:
    """
    提供与交易数据相关的业务查询服务。
    """
    def __init__(self, repository: BaseRepository):
        self.repository = repository
        print("TradingDataQuery 已初始化。")

    def get_latest_trading_dates(self, n: int = 1) -> list[pd.Timestamp]:
        """
        获取最新的 n 个交易日。

        :param n: 要获取的最新交易日数量。
        :return: 一个包含 pd.Timestamp 对象的列表，按日期降序排列。
        """
        table_name = TableName.TRADE_DATES.value
        date_column = Fields.CALENDAR_DATE.value
        is_trading_day_column = Fields.IS_TRADING_DAY.value

        print(f"[服务查询] 正在从 {table_name} 查询最新的 {n} 个交易日...")
        
        # 使用注入的 repository 来执行底层查询
        # 注意：这里我们直接使用 read_sql，因为它更灵活地支持复杂的 SELECT 语句
        try:
            query = (
                f"SELECT {date_column} FROM {table_name} "
                f"WHERE {is_trading_day_column} = 1 "
                f"ORDER BY {date_column} DESC "
                f"LIMIT {n}"
            )
            
            # 直接访问 connection 和 pandas 来执行原生 SQL
            df = pd.read_sql(query, self.repository.connection)

            if df.empty:
                print("警告: 未在数据库中找到任何交易日。")
                return []

            dates = pd.to_datetime(df[date_column]).tolist()
            print(f"成功查询到 {len(dates)} 个交易日。")
            return dates
            
        except Exception as e:
            print(f"[服务查询] 查询最新交易日时出错: {e}")
            return []

    def get_sync_status(self, symbol: str, table: TableName) -> dict | None:
        """ 获取特定股票和表的同步状态 """
        try:
            # 注意：这里的实现假设 `load` 方法可以处理更复杂的过滤
            # 我们需要确保 `load` 支持基于多个字段的过滤
            df = self.repository.load(
                TableName.SYNC_STATUS.value,
                filters={Fields.SYMBOL.value: symbol, 'table_name': table.value}
            )
            if not df.empty:
                return df.iloc[0].to_dict()
            return None
        except ValueError as e:
            # 如果表不存在，load 会抛出 ValueError
            print(f"查询同步状态时出错 (可能表不存在): {e}")
            return None

    def get_all_stocks(self) -> pd.DataFrame:
        """ 获取所有股票 """
        return self.repository.load(TableName.STOCK_LIST.value)

    def get_all_trade_dates(self) -> pd.DataFrame:
        """ 获取所有交易日 """
        return self.repository.load(TableName.TRADE_DATES.value)

    def get_all_stocks_without_indices(self) -> pd.DataFrame:
        """从 STOCK_LIST 表中加载所有非指数的股票代码。"""
        df = self.get_all_stocks()
        if df.empty:
            return pd.DataFrame()
        
        # 过滤掉指数
        df = df[~df[Fields.SYMBOL.value].str.startswith(('sh.000', 'sz.399'))]
        return df