import os
import sqlite3
import pandas as pd
import baostock as bs
import datetime
import logging
import time
import contextlib
import functools
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stock_data.db')
# DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_data.db')

# =============== 数据库链接装饰器 =================
def with_db_connection(func):
    """数据库连接的装饰器，自动处理连接的创建、提交和关闭"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        logger.info(f"数据库连接已建立: {DB_PATH}")
        result = func(conn, cursor, *args, **kwargs)
        conn.commit()
        logger.info("数据库操作已提交")
        cursor.close()
        logger.info("数据库连接已关闭")
        return result
    return wrapper

# ================= BaoStock 登录语法糖 =================

# 方式2: 装饰器
def bs_login_required(func):
    """BaoStock 登录装饰器，确保函数执行时已登录 BaoStock"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        lg = bs.login()
        logger.info("BaoStock登录成功")
        if lg.error_code != '0':
            logger.error(f"登录失败: {lg.error_msg}")
            raise ConnectionError(f"BaoStock登录失败: {lg.error_msg}")
        result = func(*args, **kwargs)
        bs.logout()
        return result
    return wrapper

# ================= 交易日数据 =================

# @bs_login_required
def fetch_trade_dates(start_date=None, end_date=None):
    """获取交易日数据"""
    if start_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"获取交易日数据: {start_date} 至 {end_date}")
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    if rs.error_code != '0':
        logger.error(f"查询交易日期失败: {rs.error_msg}")
        return None
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    # 筛选出是交易日的记录并按日期降序排序
    df = df[df['is_trading_day'] == '1']
    logger.info(f"获取到 {len(df)} 条交易日数据")
    return df['calendar_date']

# ================= 股票基本信息 =================

@bs_login_required
def fetch_stock_list(start_date=None, end_date=None):
    """获取交易日数据"""
    if start_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    logger.info(f"获取 {end_date} 的股票列表")
    rs = bs.query_all_stock(end_date)
    df = rs.get_data()
    df = df[~df['code'].str.startswith(('sh.000', 'sz.399'))]
    # df = df[df['code'].str.startswith(('sh.688', 'sz.300'))]  # 科创板和创业板
    logger.info(f"获取到 {len(df)} 只股票信息")

    return df['code'], df['code_name']

def fetch_daily_kline(code, start_date=None, end_date=None, adjustflag="2"):
    """获取股票日K线前复权数据
    
    参数:
        code: 股票代码，格式如 sh.600000
        start_date: 开始日期，默认为一年前
        end_date: 结束日期，默认为昨天
        adjustflag: 复权类型，2-前复权，1-后复权，3-不复权，默认前复权
    """
    if start_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"获取 {code} 从 {start_date} 到 {end_date} 的日K线数据")
    
    # K线数据字段
    fields = "date,open,high,low,close,volume,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM"
    
    rs = bs.query_history_k_data_plus(code, fields, 
                                      start_date=start_date, 
                                      end_date=end_date,
                                      frequency="d", 
                                      adjustflag=adjustflag)
    
    if rs.error_code != '0':
        logger.error(f"获取K线数据失败: {rs.error_msg}")
        return None
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    if not data_list:
        logger.warning(f"{code} 在指定时间段没有数据")
        return None
    
    df = pd.DataFrame(data_list, columns=rs.fields)

    # 将能转换成数值的列都转换成数值类型
    numeric_cols = []
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col])
        numeric_cols.append(col)
    df[numeric_cols] = df[numeric_cols].infer_objects(copy=False)
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    # 3. 使用 ffill 和 bfill 替代 fillna(method=...)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    logger.info(f"获取到 {len(df)} 条K线数据")
    return df

def get_table_info_from_db(conn, cursor):
    """获取数据库中已存在的K线数据表信息"""
    
    # 查询所有以line_开头的表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'line_%'")
    # cursor
    tables = cursor.fetchall()
    table_info = {}
    for table in tables:
        table_name = table[0]
        # 解析表名 line_code_startdate_enddate
        parts = table_name.split('_')
        if len(parts) == 5:
            code = parts[1]
            name = parts[2]
            start_date = parts[2]
            end_date = parts[3]
            lens = parts[4]
            table_info[code] = {
                'table_name': table_name,
                'name': name,
                'start_date': start_date,
                'end_date': end_date,
                'length': lens
            }
    return table_info

def save_kline_to_db(conn, cursor, code, name, df, start_date, end_date):
    """保存K线数据到数据库"""
    if df is None or df.empty:
        logger.warning(f"{code} 没有数据可保存")
        return None
    
    # 格式化日期为YYYYMMDD格式用于表名
    start_date_fmt = start_date.replace('-', '')
    end_date_fmt = end_date.replace('-', '')
    
    # 表名格式：line_code_startdate_enddate
    table_name = f"line_{code.replace('.', '')}_{name}_{start_date_fmt}_{end_date_fmt}_{len(df)}"
    
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    
    logger.info(f"已将 {code} 的 {len(df)} 条K线数据保存到表 {table_name}")
    return table_name

class DataFramePipeline:
    def __init__(self):
        self.plugins = []

    def register(self, plugin):
        """注册一个数据处理插件"""
        self.plugins.append(plugin)

    def run(self, df):
        """按顺序执行所有插件"""
        for plugin in self.plugins:
            df = plugin(df)
        return df

def calc_ma30(df):
    """
    计算30日收盘价滑动均线和30日成交量均值，并去除前面没有均线的行
    :param df: 包含至少 'close' 和 'volume' 列的DataFrame
    :return: 新的DataFrame，包含'ma30'和'vol_ma30'列，且去除前面没有均线的行
    """
    df['ma30'] = df['close'].rolling(window=30).mean()
    df['vol_ma30'] = df['volume'].rolling(window=30).mean()
    df = df.dropna(subset=['ma30', 'vol_ma30']).reset_index(drop=True)
    return df

def normalize_by_ma30(df):
    """
    将 open, high, low, close 分别除以30日均线（ma30），
    volume 除以30日成交量均值（vol_ma30），生成新列
    :param df: 包含 'open', 'high', 'low', 'close', 'ma30', 'volume', 'vol_ma30' 列的DataFrame
    :return: 新的DataFrame，包含归一化后的新列
    """
    for col in ['open', 'high', 'low', 'close']:
        df[f'{col}_ma30_ratio'] = df[col] / df['ma30']
    df['volume_vol_ma30_ratio'] = df['volume'] / df['vol_ma30']
    return df

# 示例用法
pipeline = DataFramePipeline()
pipeline.register(calc_ma30)
pipeline.register(normalize_by_ma30)
# df = pipeline.run(df)

@with_db_connection
@bs_login_required
def update_stock_daily_kline(conn, cursor, codes=None, force_update=False, process=False):
    """更新股票日K线数据
    
    参数:
        codes: 要更新的股票代码列表，默认为None表示更新所有A股
        force_update: 是否强制更新所有数据，默认False只更新新数据
    """
    # 获取最新的交易日作为today
    trade_dates = fetch_trade_dates()
    if trade_dates is None or trade_dates.empty:
        logger.error("无法获取交易日数据")
        return
    today = trade_dates.iloc[-1]
    
    # 获取数据库中已有的表信息
    existing_tables = get_table_info_from_db(conn, cursor)
    
    # 如果没有指定股票代码，则获取所有A股
    if codes is None:
        codes, names = fetch_stock_list()
        if codes is None:
            logger.error("无法获取股票列表，更新失败")
            return
    
    total_stocks = len(codes)
    logger.info(f"开始更新 {total_stocks} 只股票的K线数据")
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, (code, name) in enumerate(zip(codes, names)):
        
        code_clean = code.replace('.', '')
            
        # 检查是否已存在该股票的数据表
        if code_clean in existing_tables and not force_update:
            existing_info = existing_tables[code_clean]
            existing_end_date = existing_info['end_date']
                
            # 将YYYYMMDD格式转换为YYYY-MM-DD
            formatted_end_date = f"{existing_end_date[:4]}-{existing_end_date[4:6]}-{existing_end_date[6:8]}"
                
            # 如果已有数据的结束日期是今天，则跳过
            if formatted_end_date == today:
                logger.info(f"[{i+1}/{total_stocks}] {code} 数据已是最新，跳过")
                skip_count += 1
                continue
                
            # 获取从已有数据结束日期到今天的新数据
            df = fetch_daily_kline(code, start_date=formatted_end_date, end_date=today)
            if process:
                # 如果需要处理数据，则使用管道处理
                df = pipeline.run(df)

            if df is not None and not df.empty:
                # 读取已有数据
                existing_df = pd.read_sql_query(f"SELECT * FROM {existing_info['table_name']}", conn)
                    
                # 合并新旧数据，去重
                merged_df = pd.concat([existing_df, df]).drop_duplicates(subset=['date'], keep='last')
                    
                # 获取合并后的数据范围
                new_start_date = merged_df['date'].min()
                new_end_date = merged_df['date'].max()
                    
                # 保存合并后的数据，并使用新的表名
                save_kline_to_db(conn, cursor, code, name, merged_df, new_start_date, new_end_date)
                    
                cursor.execute(f"DROP TABLE IF EXISTS {existing_info['table_name']}")
                    
                logger.info(f"[{i+1}/{total_stocks}] {code} 数据更新成功，从 {existing_end_date} 更新到 {new_end_date.replace('-', '')}")
                success_count += 1
            else:
                logger.warning(f"[{i+1}/{total_stocks}] {code} 无法获取新数据")
                fail_count += 1
        else:
            # 获取一年的数据
            start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
            df = fetch_daily_kline(code, start_date=start_date, end_date=today)
            if df is not None and not df.empty:
                save_kline_to_db(conn, cursor, code, name, df, start_date, today)
                logger.info(f"[{i+1}/{total_stocks}] {code} 新增数据成功")
                success_count += 1
            else:
                logger.warning(f"[{i+1}/{total_stocks}] {code} 无法获取数据")
                fail_count += 1
    
    logger.info(f"K线数据更新完成: 成功 {success_count}, 失败 {fail_count}, 跳过 {skip_count}")

if __name__ == "__main__":
    # rs = fetch_stock_list()
    # print(rs)
    # print(fetch_trade_dates())
    # update_stock_daily_kline()
    print(DB_PATH)
