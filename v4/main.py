
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
from src.common.enums import DatabaseConfig, TableName # 导入新的枚举

# --- 全局配置 (现在由 schema.py 管理) ---
# BAOSTOCK_DB_PATH = 'd:/stock/v4/baostock_meta.db'
# KLINE_DB_PATH = 'd:/stock/v4/kline_data.db'
# STOCK_LIST_TABLE = 'stock_list'
# TRADE_DATES_TABLE = 'trade_dates'
# KLINE_5MIN_TABLE = 'kline_5min'
# KLINE_DATA_RANGE_TABLE = 'kline_data_range'

# --- 模块化功能 ---

def update_data_range(repo: SqliteRepository, code: str, kline_df: pd.DataFrame):
    """
    更新或创建股票在 kline_data_range 表中的数据时间范围记录。
    """
    if kline_df.empty:
        return

    table_name = TableName.KLINE_DATA_RANGE
    new_start_dt = kline_df.index.min()
    new_end_dt = kline_df.index.max()

    try:
        existing_range_df = repo.load(table_name=table_name, filters={'code': code})

        if existing_range_df.empty:
            print(f"为 {code} 创建新的数据范围记录...")
            range_df = pd.DataFrame([{
                'code': code,
                'start_datetime': new_start_dt,
                'end_datetime': new_end_dt
            }])
            repo.save(range_df, table_name=table_name, if_exists='append', save_index=False)
        else:
            old_start_dt = pd.to_datetime(existing_range_df['start_datetime'].iloc[0])
            old_end_dt = pd.to_datetime(existing_range_df['end_datetime'].iloc[0])
            
            update_payload = {}
            if new_start_dt < old_start_dt:
                update_payload['start_datetime'] = new_start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')
            if new_end_dt > old_end_dt:
                update_payload['end_datetime'] = new_end_dt.strftime('%Y-%m-%d %H:%M:%S.%f')

            if update_payload:
                set_clauses = ", ".join([f"{key} = ?" for key in update_payload.keys()])
                params = list(update_payload.values()) + [code]
                
                print(f"为 {code} 更新数据范围记录: {update_payload}...")
                cursor = repo.connection.cursor()
                cursor.execute(
                    f"UPDATE {table_name} SET {set_clauses} WHERE code = ?",
                    params
                )
                repo.connection.commit()
    except Exception as e:
        print(f"错误: 更新数据范围表 for {code} 时出错: {e}")

def update_base_info():
    """
    从 BaoStock 获取最新的股票列表和交易日历，并存入本地元数据数据库。
    股票列表将基于最新的前一个交易日获取，以保证数据完整性。
    """
    print("--- [任务1] 开始更新基础信息 (股票列表和交易日) ---")
    try:
        with BaoStockConnector() as bs_connector, SqliteRepository(db_path=DatabaseConfig.META) as repo:
            
            # 1. 获取并存储最新的交易日历
            print("正在获取交易日历...")
            end_date = datetime.now().strftime('%Y-%m-%d')
            # 我们获取一个较长的时间窗口以确保能找到前一个交易日
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d') 
            trade_dates_df = bs_connector.get_trade_dates(start_date=start_date, end_date=end_date)
            
            if not trade_dates_df.empty:
                repo.save(trade_dates_df, table_name=TableName.TRADE_DATES, if_exists='replace', save_index=False)
                print(f"成功存储 {len(trade_dates_df)} 条交易日信息到 <{TableName.TRADE_DATES}> 表。")

                # 2. 找到倒数第二个交易日
                trading_days = trade_dates_df[trade_dates_df['is_trading_day'] == 1]['calendar_date'].tolist()
                if len(trading_days) >= 2:
                    target_date = trading_days[-2]
                    print(f"将使用前一个交易日 ({target_date}) 的数据来获取股票列表。")
                    
                    # 3. 使用目标日期获取股票列表
                    print("正在获取股票列表...")
                    stock_list_df = bs_connector.get_stock_list(date=target_date)
                    if not stock_list_df.empty:
                        repo.save(stock_list_df, table_name=TableName.STOCK_LIST, if_exists='replace', save_index=False)
                        print(f"成功存储 {len(stock_list_df)} 条股票信息到 <{TableName.STOCK_LIST}> 表。")
                    else:
                        print(f"警告: 未能获取到 {target_date} 的股票列表。")
                else:
                    print("警告: 交易日数量不足，无法确定前一个交易日，暂时不更新股票列表。")
            else:
                print("警告: 未能获取到交易日历，无法更新股票列表。")

    except Exception as e:
        print(f"错误: 更新基础信息时出错: {e}")
    finally:
        print("--- [任务1] 基础信息更新完成 ---\n")


def update_kline_data(stock_codes: list, start_date_default: str = '2023-01-01'):
    """
    为指定的股票列表增量更新5分钟K线数据。
    """
    print(f"--- [任务2] 开始为 {len(stock_codes)} 只股票增量更新5分钟K线数据 ---")
    
    try:
        with BaoStockConnector() as bs_connector, SqliteRepository(db_path=DatabaseConfig.KLINE) as repo:
            total_stocks = len(stock_codes)
            for i, code in enumerate(stock_codes):
                print(f"\n[{i+1}/{total_stocks}] 正在处理股票: {code}")
                
                start_datetime_str = start_date_default
                try:
                    range_info_df = repo.load(table_name=TableName.KLINE_DATA_RANGE, filters={'code': code})
                    if not range_info_df.empty:
                        latest_timestamp = pd.to_datetime(range_info_df['end_datetime'].iloc[0])
                        start_datetime = latest_timestamp + timedelta(minutes=1)
                        start_datetime_str = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
                        print(f"从范围表得知本地最新数据时间: {latest_timestamp}，将从 {start_datetime_str} 开始更新。")
                    else:
                        print(f"范围表中无 {code} 的数据，将从默认日期 {start_date_default} 开始下载。")
                except Exception:
                    try:
                        latest_entry_df = repo.load(
                            table_name=TableName.KLINE_5MIN,
                            filters={'code': code},
                            order_by='datetime DESC',
                            limit=1
                        )
                        if not latest_entry_df.empty:
                            latest_timestamp = pd.to_datetime(latest_entry_df.index[0])
                            start_datetime = latest_timestamp + timedelta(minutes=1)
                            start_datetime_str = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
                            print(f"从K线大表得知本地最新数据时间: {latest_timestamp}，将从 {start_datetime_str} 开始更新。")
                        else:
                            print(f"K线大表中亦无 {code} 的数据，将从默认日期 {start_date_default} 开始下载。")
                    except Exception as e:
                        print(f"查询K线大表时发生错误 (可能表不存在): {e}。将从默认日期 {start_date_default} 开始下载。")

                end_datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                start_date = start_datetime_str.split(' ')[0]
                end_date = end_datetime_str.split(' ')[0]

                try:
                    kline_df = bs_connector.get_kline_data(
                        code=code,
                        start_date=start_date,
                        end_date=end_date,
                        frequency="5"
                    )
                    
                    if 'start_datetime' in locals() and isinstance(locals().get('start_datetime'), pd.Timestamp):
                        kline_df = kline_df[kline_df.index >= start_datetime]

                    if not kline_df.empty:
                        print(f"成功获取 {len(kline_df)} 条新的K线数据。")
                        repo.save(
                            df=kline_df,
                            table_name=TableName.KLINE_5MIN,
                            if_exists='append',
                            save_index=True
                        )
                        print(f"已将新数据追加到 <{TableName.KLINE_5MIN}> 表。")

                        update_data_range(repo, code, kline_df)
                    else:
                        print("没有获取到新的K线数据。")

                except ValueError as e:
                    print(f"获取K线数据时出错: {e}")
                
                time.sleep(0.1)

    except Exception as e:
        print(f"错误: 更新K线数据时发生严重错误: {e}")
    finally:
        print("\n--- [任务2] K线数据更新流程结束 ---")

def main():
    """
    主函数，协调所有数据更新任务。
    """
    update_base_info()

    print("\n--- 准备执行K线更新 ---")
    try:
        with SqliteRepository(db_path=DatabaseConfig.META) as repo:
            stock_list_df = repo.load(
                table_name=TableName.STOCK_LIST,
                filters={"code_name NOT LIKE '%B股%' AND code LIKE 'sh.6%' OR code LIKE 'sz.0%' OR code LIKE 'sz.3%'"}
            )
            
            if not stock_list_df.empty:
                stock_codes = stock_list_df['code'].tolist()
                print(f"将为 {len(stock_codes)} 只A股进行K线更新。")
                
                update_kline_data(stock_codes)
            else:
                print("未能从本地数据库加载到股票列表，无法进行K线更新。")

    except Exception as e:
        print(f"错误: 在主流程中加载股票列表时出错: {e}")


if __name__ == '__main__':
    main()