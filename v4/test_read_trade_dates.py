
import pandas as pd
from src.data.repositories.sqlite_repository import SqliteRepository
from src.common.enums import DatabaseConfig, TableName

def view_trade_dates():
    """
    使用 SqliteRepository 从数据库加载并显示 trade_dates 表的内容。
    """
    db_path = DatabaseConfig.STOCK_DB.value
    table_name = TableName.TRADE_DATES.value

    print(f"--- 开始从数据库 {db_path} 读取表 {table_name} ---")
    
    try:
        with SqliteRepository(db_path) as repo:
            # 加载整个表
            trade_dates_df = repo.load(table_name)
            
            if not trade_dates_df.empty:
                print("\n[+] 数据加载成功!\n")
                
                # 设置 pandas 显示选项以完整显示内容
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)
                
                print("--- 表信息 (Info) ---")
                trade_dates_df.info()
                
                print("\n--- 数据头部 (Head) ---")
                print(trade_dates_df.head())
                
                print("\n--- 数据尾部 (Tail) ---")
                print(trade_dates_df.tail())
            else:
                print("[-] 表为空或不存在。")

    except Exception as e:
        print(f"[-] 读取数据时发生错误: {e}")

    print("\n--- 读取任务结束 ---")

if __name__ == "__main__":
    view_trade_dates()