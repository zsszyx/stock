from datetime import date, timedelta
import pandas as pd
from src.data.managers.base import BaseManager
from src.common.enums import TableName, Fields

class TradeDatesManager(BaseManager):
    """
    交易日历数据管理器。
    负责同步和持久化交易日历数据，支持增量更新。
    """

    def sync_data(self, start_date: str, end_date: str):
        """
        同步指定日期范围内的交易日历数据，并自动处理增量更新。

        :param start_date: 期望的开始日期，格式 'YYYY-MM-DD'。
        :param end_date: 结束日期，格式 'YYYY-MM-DD'。
        """
        table_name = TableName.TRADE_DATES.value
        date_column = Fields.CALENDAR_DATE

        # 1. 查询数据库中最新的日期，以确定增量更新的起点
        latest_date_in_db = self.repository.get_latest_date(table_name, date_column)

        effective_start_date = start_date
        if_exists_strategy = 'replace'
        save_index = True

        if latest_date_in_db:
            # 如果数据库中已有数据，则从最新日期的后一天开始同步
            effective_start_date = (latest_date_in_db + timedelta(days=1)).strftime('%Y-%m-%d')
            if_exists_strategy = 'append'
            save_index = False # 追加模式下不保存索引，因为索引是自动生成的
            print(f"数据库中最新交易日为: {latest_date_in_db.strftime('%Y-%m-%d')}")
            print(f"将从 {effective_start_date} 开始增量更新。")
        else:
            print("未在数据库中找到交易日历数据，将进行全量同步。")

        # 如果计算出的开始日期已经晚于或等于结束日期，则无需同步
        if pd.to_datetime(effective_start_date) >= pd.to_datetime(end_date):
            print(f"数据已是最新 (截止到 {end_date})，无需同步。")
            return

        print(f"--- 开始同步交易日历数据 ({effective_start_date} to {end_date}) ---")
        
        # 2. 从 Connector 获取数据
        trade_dates_df = self.connector.get_trade_dates(effective_start_date, end_date)
        
        if trade_dates_df.empty:
            print("未获取到新的交易日历数据，任务结束。")
            return

        # 3. 将数据存入 Repository
        # 关键：使用 'append' 模式来增加新数据，而不是替换整个表
        self.repository.save(trade_dates_df, table_name, if_exists=if_exists_strategy, save_index=save_index)
        
        print(f"--- 交易日历数据同步完成 ---")