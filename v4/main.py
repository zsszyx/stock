
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# 动态调整 Python 路径，以便能够导入 src 目录下的模块
# 这是在项目根目录运行脚本时确保模块可被发现的常见做法
import sys
# 将 v4 目录的父目录（即 d:\stock）添加到 sys.path
# 这样 Python 解释器就能找到 v4.src.data...
# 注意：这里假设脚本是从 d:\stock\v4 目录或者其父目录执行
# 为了更好的健壮性，我们计算出脚本所在的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..'))
# sys.path.insert(0, project_root)
# 暂时使用硬编码的路径，后续可以优化
sys.path.insert(0, 'd:\\stock\\v4')


from src.data.connectors.baostock_connector import BaoStockConnector
from src.data.repositories.sqlite_repository import SqliteRepository
from src.data.managers.trade_dates_manager import TradeDatesManager
from src.common.enums import DatabaseConfig, TableName, Fields # 导入新的枚举


def sync_trade_dates(start_date: str, end_date: str):
    """
    使用 TradeDatesManager 同步指定时间范围的交易日历。
    """
    print("--- [任务1] 开始更新交易日历 ---")
    try:
        with BaoStockConnector() as connector, SqliteRepository(db_path=DatabaseConfig.STOCK_DB) as repo:
            manager = TradeDatesManager(connector, repo)
            manager.sync_data(start_date, end_date)
    except Exception as e:
        print(f"错误: 同步交易日历时出错: {e}")
    finally:
        print("--- [任务1] 交易日历更新完成 ---\n")

def main():
    """
    主函数，协调所有数据更新任务。
    """
    # 1. 首先，同步最近一年的交易日历
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    sync_trade_dates(start_date, end_date)


if __name__ == '__main__':
    main()