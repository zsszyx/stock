import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys

sys.path.insert(0, 'd:\\stock\\v4')

from src.data.connectors.baostock_connector import BaoStockConnector
from src.data.repositories.sqlite_repository import SqliteRepository
from src.data.managers.trade_dates_manager import TradeDatesManager
from src.data.managers.stock_list_manager import StockListManager
from src.data.managers.kline_manager import KLineManager
from src.data.queries.trading_data_query import TradingDataQuery
from src.common.enums import DatabaseConfig

def sync_trade_dates(connector, repo):
    """
    使用 TradeDatesManager 同步交易日历。
    """
    print("--- [任务1] 开始更新交易日历 ---")
    try:
        manager = TradeDatesManager(connector, repo)
        manager.sync_data()
    except Exception as e:
        print(f"错误: 同步交易日历时出错: {e}")
    finally:
        print("--- [任务1] 交易日历更新完成 ---\n")

def sync_stock_list(connector, repo):
    """
    使用 StockListManager 同步最新的股票列表。
    """
    print("--- [任务2] 开始更新股票列表 ---")
    try:
        query_service = TradingDataQuery(repo)
        manager = StockListManager(connector, repo, query_service)
        manager.sync_data()
    except Exception as e:
        print(f"错误: 同步股票列表时出错: {e}")
    finally:
        print("--- [任务2] 股票列表更新完成 ---\n")

def sync_kline_data(connector, repo):
    """
    使用 KLineManager 同步5分钟K线数据。
    """
    print("--- [任务3] 开始更新5分钟K线数据 ---")
    try:
        query_service = TradingDataQuery(repo)
        manager = KLineManager(connector, repo, query_service)
        manager.sync_data()
    except Exception as e:
        print(f"错误: 同步5分钟K线数据时出错: {e}")
    finally:
        print("--- [任务3] 5分钟K线数据更新完成 ---\n")

def main():
    """
    主函数，协调所有数据更新任务。
    """
    with BaoStockConnector() as connector, SqliteRepository(db_path=DatabaseConfig.STOCK_DB) as repo:
        # 1. 首先，同步交易日历
        sync_trade_dates(connector, repo)

        # 2. 然后，同步股票列表
        sync_stock_list(connector, repo)

        # 3. 最后，同步5分钟K线数据
        sync_kline_data(connector, repo)

if __name__ == '__main__':
    main()