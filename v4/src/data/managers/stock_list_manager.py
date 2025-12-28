from src.data.managers.base import BaseManager
from src.common.enums import TableName, COLUMN_ALIASES
from src.data.queries.trading_data_query import TradingDataQuery

class StockListManager(BaseManager):
    """
    管理和同步股票列表数据。
    """
    def __init__(self, connector, repository, trading_data_query: TradingDataQuery):
        super().__init__(connector, repository)
        self.trading_data_query = trading_data_query
    def sync_data(self, *args, **kwargs):
        """
        同步股票列表的核心逻辑。

        该方法会获取最新的交易日，并根据该交易日的前一个交易日来获取股票列表，
        以确保数据的完整性和准确性。
        """
        print("--- [任务] 开始更新股票列表 ---")
        
        # 1. 从数据库获取最新的两个交易日
        #    我们获取两个是为了找到“前一个”交易日
        latest_dates = self.trading_data_query.get_latest_trading_dates(n=2)
        
        if not latest_dates or len(latest_dates) < 2:
            print("错误: 数据库中的交易日数据不足 (需要至少2个交易日)，无法确定用于同步的日期。")
            print("--- [任务] 股票列表更新失败 ---")
            return

        # 2. 确定用于同步的日期 (最新的交易日的前一个交易日)
        #    latest_dates 已经是按降序排列的，所以第二个元素就是前一个交易日
        sync_date = latest_dates[1]
        sync_date_str = sync_date.strftime('%Y-%m-%d')
        print(f"确定用于同步股票列表的日期为: {sync_date_str}")

        # 3. 使用 Connector 获取该日期的股票列表
        print(f"正在从远端获取 {sync_date_str} 的股票列表...")
        stock_list_df = self.connector.get_stock_list(sync_date_str)

        if stock_list_df.empty:
            print(f"警告: 在日期 {sync_date_str} 未获取到股票列表数据。")
            print("--- [任务] 股票列表更新完成 (无数据) ---")
            return

        # 4. 标准化列名并使用 Repository 将数据保存到数据库
        #    在保存前，先将列名标准化 (e.g., 'code' -> 'symbol')
        stock_list_df.rename(columns={k: v.value for k, v in COLUMN_ALIASES.items() if k in stock_list_df.columns}, inplace=True)

        table_name = TableName.STOCK_LIST.value
        self.repository.save_data(stock_list_df, table_name, if_exists='replace')
        
        print(f"已成功将 {len(stock_list_df)} 条股票列表数据保存到表 {table_name} 中。")
        print("--- [任务] 股票列表更新完成 ---")