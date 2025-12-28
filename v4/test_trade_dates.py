
import sys
from datetime import datetime, timedelta
# 暂时使用硬编码的路径，后续可以优化
sys.path.insert(0, 'd:\\stock\\v4')

from src.data.connectors.baostock_connector import BaoStockConnector

def test_get_trade_dates():
    """
    测试 BaoStockConnector.get_trade_dates() 方法。
    """
    print("--- 开始测试 get_trade_dates ---")
    try:
        with BaoStockConnector() as connector:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            print(f"正在获取从 {start_date} 到 {end_date} 的交易日历...")
            trade_dates_df = connector.get_trade_dates(start_date=start_date, end_date=end_date)
            
            if not trade_dates_df.empty:
                print("\n--- 获取结果 ---")
                print("数据头部:")
                print(trade_dates_df.head())
                print("\n数据尾部:")
                print(trade_dates_df.tail())
                print("\n数据信息:")
                trade_dates_df.info()
            else:
                print("未能获取到任何交易日数据。")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    finally:
        print("\n--- 测试结束 ---")

if __name__ == '__main__':
    test_get_trade_dates()