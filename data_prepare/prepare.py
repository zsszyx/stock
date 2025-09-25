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
import akshare as ak

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stock_data.db')
# DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_data.db')

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

@with_db_connection
def get_industry_data(conn, cusor):
    sql = "select * from stock_industry"
    df = pd.read_sql(sql, conn)
    # 把code从sh.000001 转换为 sh000001
    df['code'] = df['code'].str.replace('sh.', 'sh')
    # 把code从sz.000001 转换为 sz000001
    df['code'] = df['code'].str.replace('sz.', 'sz')
    return df[['code', 'industry']]

def get_table_info_from_db(conn, cursor):
    """获取数据库中已存在的K线数据表信息"""  
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND (name LIKE 'daily_%' OR name LIKE 'minute60_%')")
    # cursor
    tables = cursor.fetchall()
    tables = [table[0] for table in tables]
    table_info = {}
    for table in tables:
        table_name = table
        # 解析表名 line_code_startdate_enddate
        parts = table_name.split('_')
        if len(parts) == 6:
            freq = parts[0]
            code = parts[1]
            name = parts[2]
            start_date = parts[3]
            end_date = parts[4]
            lens = parts[5]
            table_info[code] = {
                'freq': freq,
                'table_name': table_name,
                'name': name,
                'start_date': start_date,
                'end_date': end_date,
                'length': lens
            }
    if not table_info:
        logger.error("数据库中没有找到K线数据表")
        return
    return table_info

@with_db_connection
def get_stocks_latest_data(conn, cursor, freq='daily', length=30):
    """
    从数据库中读取指定股票的最后length行数据
    
    参数:
        stock_codes: 股票代码列表，如 ['sh.600000', 'sz.000001']
        freq: 数据频率，'daily' 或 'minute60'
        length: 要读取的最后几行数据
    
    返回:
        dict: {股票代码: DataFrame} 格式的字典
    """
    # 获取数据库中指定频率的所有表信息
    existing_tables = get_table_info_from_db(conn, cursor)
    freq_tables = {k: v for k, v in existing_tables.items() if v['freq'] == freq}
    
    if not freq_tables:
        raise ValueError(f"数据库中没有找到任何{freq}数据表")
    
    logger.info(f"开始读取 {len(freq_tables)} 只指定股票的最后 {length} 行 {freq} 数据")
    
    stock_data = {}
    success_count = 0
    fail_count = 0

    for i, (code, _) in enumerate(freq_tables.items()):
        # 将股票代码转换为数据库中的格式
        code_clean = code.replace('.', '')
        
        table_info = freq_tables[code_clean]
        table_name = table_info['table_name']
        stock_name = table_info['name']
        
        try:
            # 构造查询语句，获取最后length行数据
            query = f"""
            SELECT * FROM {table_name} 
            ORDER BY date DESC 
            LIMIT {length}
            """
            
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                logger.warning(f"[{i+1}/{len(freq_tables)}] {code_clean} 表 {table_name} 中没有数据")
                fail_count += 1
                continue
            
            # 按日期正序排列
            df = df.sort_values('date').reset_index(drop=True)
            # print(df.columns)
            
            stock_data[code] = {
                'code': code,
                'name': stock_name,
                'data': df
            }

            logger.info(f"[{i+1}/{len(freq_tables)}] {code} ({stock_name}) 读取成功，获得 {len(df)} 行数据")
            success_count += 1
            
        except Exception as e:
            logger.error(f"[{i+1}/{len(freq_tables)}] {code} 读取数据失败: {str(e)}")
            fail_count += 1
            continue
    
    logger.info(f"指定股票数据读取完成: 成功 {success_count}, 失败 {fail_count}")
    return stock_data

def get_stock_merge_table(length=30, freq='daily'):
    """
    获取所有股票的最新length条数据，拼接为一个大表。
    - 只保留包含'market_value'列的股票数据
    - 检查所有DataFrame列名是否一致
    - 检查每列的null和0比例不超过5%，并用bfill和fillna填充
    - 最终按code和date拼接所有股票历史行情
    返回：拼接后的DataFrame
    """
    stock_data_dict = get_stocks_latest_data(freq=freq, length=length)

    # 从dict中取出'sh000001'为key的项，并删除
    sh000001_item = stock_data_dict.pop('sh000001')

    sh000001_df = sh000001_item['data'][['date','close']].copy()
    sh000001_df = sh000001_df.rename(columns={'close': 'sh_close'})

    dflist = []
    for idx, (code, info) in enumerate(stock_data_dict.items()):
        logger.info(f"merge股票 {code} 进度 {idx+1}/{len(stock_data_dict)} \r")
        df = info['data']
        # 必须包含指定的关键列，便于扩展
        required_columns = ['market_value','date','open','high','low','close','volume','turn','amount','pctChg']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"{code} 数据缺失关键列，已跳过")
            continue
        df['code'] = code
        dflist.append(df)
    if not dflist:
        raise ValueError("没有符合条件的股票数据")

    # 拼接所有股票
    merged = pd.concat(dflist, axis=0, ignore_index=True)

    # 除涨停
    merged = exclude_limit_up_down(merged)

    # 合并上'sh000001'的pctChg列
    merged = pd.merge(merged, sh000001_df, on='date', how='left')
    # 按code, date排序
    merged = merged.sort_values(['code', 'date']).reset_index(drop=True)

    merged = calculate_amount_ratio(merged)
    return merged

# 获取和行业信息合并后的大表
def get_stock_merge_industry_table(length=30, freq='daily'):
    """
    获取所有股票的最新length条数据，拼接为一个大表，
    并合并上行业信息。
    """
    # 获取股票数据
    stock_df = get_stock_merge_table(length=length, freq=freq)

    # 获取行业信息
    industry_df = get_industry_data()

    # 合并数据
    merged = pd.merge(stock_df, industry_df, on='code', how='left')

    return merged

# 标记涨跌停
def exclude_limit_up_down(df):
    # 复制所有列，加上real_前缀
    cols = []
    for col in df.columns:
        if col not in ['code', 'date', 'market_value']:
            df[f'real_{col}'] = df[col]
            cols.append(col)

    df['limited'] = np.where(abs(df['pctChg']) >= 9.85, True, False)
    df.loc[df['limited'], cols] = np.nan

    return df

def calculate_amount_ratio(df: pd.DataFrame):
    """
    在每个日期截面，计算每只股票的成交额(amount)占当天总成交额的比例。

    参数:
        df: 包含 'date' 和 'amount' 列的DataFrame。

    返回:
        带有 'amount_ratio' 列的DataFrame。
    """
    # 按日期分组，计算每日的总成交额
    daily_total_amount = df.groupby('date')['amount'].transform('sum')
    
    # 计算每只股票的成交额占比
    df['amount_ratio'] = df['amount'] / daily_total_amount
    return df

if __name__ == "__main__":
    # 测试获取合并表
    df = get_stock_merge_industry_table(length=60, freq='daily')
    print(df.head())
    print(df.tail())
    print(df.info())