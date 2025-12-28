
import pandas as pd
from datetime import datetime, timedelta
from .base import BaseManager
from src.data.connectors.baostock_connector import BaoStockConnector
from src.data.repositories.base import BaseRepository
from src.data.queries.trading_data_query import TradingDataQuery
from src.common.enums import TableName, Fields, SyncStatus

class KLineManager(BaseManager):
    """
    负责同步、清洗和管理5分钟K线数据的管理器。
    """

    def __init__(self, connector: BaoStockConnector, repository: BaseRepository, query: TradingDataQuery):
        """
        初始化 KLineManager。

        :param connector: BaoStockConnector 实例。
        :param repository: BaseRepository 实例。
        :param query: TradingDataQuery 实例。
        """
        super().__init__(connector, repository)
        self.query = query

    def sync_data(self):
        """
        核心方法：同步所有股票的5分钟K线数据。
        """
        print("开始同步5分钟K线数据...")

        # 1. 获取所有股票和交易日历
        all_stocks = self.query.get_all_stocks_without_indices()
        trade_dates = self.query.get_all_trade_dates()
        
        if all_stocks.empty:
            print("股票列表为空，无法进行K线同步。")
            return
            
        if trade_dates.empty:
            print("交易日历为空，无法进行K线同步。")
            return

        # 将交易日历的日期转换为集合以便快速查找
        trade_dates_set = set(pd.to_datetime(trade_dates[Fields.CALENDAR_DATE.value]).dt.date)
        min_trade_date = trade_dates[Fields.CALENDAR_DATE.value].min()
        today_str = datetime.now().strftime('%Y-%m-%d')

        # 2. 循环处理每只股票
        for index, stock in all_stocks.iterrows():
            symbol = stock[Fields.SYMBOL.value]
            print(f"--- 开始处理股票: {symbol} ---")

            # 3. 确定同步日期范围
            sync_status = self.query.get_sync_status(symbol, TableName.KLINE_5MIN)
            
            start_date = min_trade_date
            if sync_status and sync_status[Fields.SYNC_END_DATE.value]:
                last_sync_date = datetime.strptime(sync_status[Fields.SYNC_END_DATE.value], '%Y-%m-%d').date()
                start_date = (last_sync_date + timedelta(days=1)).strftime('%Y-%m-%d')

            end_date = today_str
            
            if start_date is None and end_date is None:
                print(f"无法为 {symbol} 确定同步日期范围，跳过。")
                continue

            print(f"同步日期范围: {start_date} -> {end_date}")

            # 如果开始日期在结束日期之后，说明数据已是最新，跳过
            if start_date > end_date:
                print(f"数据已是最新 (开始日期 {start_date} > 结束日期 {end_date})，跳过同步。")
                self._update_sync_status(symbol, start_date, end_date, SyncStatus.NO_DATA, "数据已是最新")
                continue

            # 4. 获取K线数据
            try:
                kline_data = self.connector.get_kline_data(
                    code=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    frequency="5" # 5分钟线
                )
            except ValueError as e:
                print(f"获取 {symbol} 的K线数据时出错: {e}")
                self._update_sync_status(symbol, start_date, end_date, SyncStatus.NO_DATA, "获取数据失败")
                continue

            if kline_data.empty:
                print(f"未找到 {symbol} 在指定期间的K线数据。")
                self._update_sync_status(symbol, start_date, end_date, SyncStatus.NO_DATA, "无新数据")
                continue
            
            # 5. 清洗数据：移除所有非交易日的数据点
            kline_data['date_only'] = kline_data[Fields.DT.value].dt.date
            cleaned_data = kline_data[kline_data['date_only'].isin(trade_dates_set)].copy()
            cleaned_data.drop(columns=['date_only'], inplace=True)

            if cleaned_data.empty:
                print(f"在 {start_date} 到 {end_date} 期间, {symbol} 没有位于交易日的数据。")
                self._update_sync_status(symbol, start_date, end_date, SyncStatus.NO_DATA, "无有效交易日数据")
                continue

            # 6. 检查数据完整性
            check_cols = [Fields.OPEN.value, Fields.HIGH.value, Fields.LOW.value, Fields.CLOSE.value, Fields.VOL.value, Fields.AMT.value]
            is_complete = not cleaned_data[check_cols].isnull().values.any()
            status = SyncStatus.DATA_COMPLETE if is_complete else SyncStatus.DATA_INCOMPLETE
            
            # 7. 保存数据到数据库
            try:
                self.repository.save_data(cleaned_data, TableName.KLINE_5MIN.value)
                print(f"成功保存 {len(cleaned_data)} 条 {symbol} 的K线数据。")
                
                # 8. 更新同步状态
                self._update_sync_status(symbol, start_date, end_date, status, "同步成功")

            except Exception as e:
                print(f"保存 {symbol} 的K线数据时出错: {e}")
                self._update_sync_status(symbol, start_date, end_date, SyncStatus.DATA_INCOMPLETE, f"数据库保存失败: {e}")

            print(f"--- 完成处理股票: {symbol} ---")

        print("所有股票的5分钟K线数据同步完成。")

    def _update_sync_status(self, symbol: str, table_name: str, status: SyncStatus, start_date: str = None, end_date: str = None, message: str = ""):
        """更新同步状态。只有成功时才记录日期。"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 只有成功状态才记录日期，其他状态将日期置为None
        if status != SyncStatus.SUCCESS:
            start_date = None
            end_date = None

        status_data = {
            Fields.SYMBOL.value: [symbol],
            Fields.TABLE_NAME.value: [table_name],
            Fields.SYNC_STATUS.value: [status.value],
            Fields.LAST_SYNC_DATE.value: [now],
            Fields.SYNC_START_DATE.value: [start_date],
            Fields.SYNC_END_DATE.value: [end_date],
            Fields.MESSAGE.value: [message]
        }
        status_df = pd.DataFrame(status_data)
        self.repository.upsert_data(status_df, TableName.SYNC_STATUS)
        logger.info(f"更新状态 for {symbol}: {status.value} ({message})")