
import sys
import os
from datetime import datetime, timedelta

# Add project root to Python path to allow imports from src
# This makes the script runnable from anywhere
try:
    # This will work if the script is in v4
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # This will work if the script is in the root
    v4_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'v4'))
    if v4_path not in sys.path:
        sys.path.insert(0, v4_path)
except NameError:
    # Fallback for interactive environments
    sys.path.insert(0, 'd:\\stock\\v4')


from src.data.connectors.baostock_connector import BaoStockConnector

def test_fetch_kline():
    """
    测试从 BaoStock 获取5分钟K线数据。
    """
    print("--- 开始测试获取5分钟K线数据 ---")
    
    # 选择一个测试股票和时间范围
    stock_code = "sh.600519"  # 贵州茅台
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')

    print(f"测试股票: {stock_code}")
    print(f"时间范围: {start_date_str} to {end_date_str}")

    try:
        with BaoStockConnector() as connector:
            kline_df = connector.get_kline_data(
                code=stock_code,
                start_date=start_date_str,
                end_date=end_date_str,
                frequency="5",  # 5分钟线
                adjust_flag="3"  # 不复权
            )

            if not kline_df.empty:
                print("\n成功获取K线数据！")
                print("数据预览 (前5条):")
                print(kline_df.head())
                print("\n数据预览 (后5条):")
                print(kline_df.tail())
                print(f"\n总共获取了 {len(kline_df)} 条数据。")
                print("数据类型信息:")
                kline_df.info()
            else:
                print("\n未能获取到任何K线数据，请检查网络或BaoStock服务状态。")

    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
    finally:
        print("\n--- 测试结束 ---")

if __name__ == "__main__":
    test_fetch_kline()