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

"""
日线表格名称格式：daily_<code>_<start_date>_<end_date>_<length>
60分钟线表格名称格式：minute60_<code>_<start_date>_<end_date>_<length>
"""

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
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
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

# @bs_login_required
def fetch_stock_list(start_date=None, end_date=None):
    """获取交易日数据"""
    if start_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = (datetime.datetime.now() - datetime.timedelta(days=20)).strftime('%Y-%m-%d')
    
    logger.info(f"获取 {end_date} 的股票列表")
    rs = bs.query_all_stock(end_date)
    df = rs.get_data()
    df = df[~df['code'].str.startswith(('sh.000', 'sz.399'))]
    df = df[~df['code_name'].str.contains(r'\*|ST|S|退')]
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

def fetch_minute60_kline(code, start_date=None, end_date=None, adjustflag="3"):
    """获取股票60分钟K线数据

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

    logger.info(f"获取 {code} 从 {start_date} 到 {end_date} 的分钟K线数据")

    # K线数据字段
    fields = "date,open,high,low,close,volume"

    rs = bs.query_history_k_data_plus(code, fields,
                                      start_date=start_date,
                                      end_date=end_date,
                                      frequency="60",
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

    logger.info(f"获取到 {len(df)} 条分钟K线数据")
    return df

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
    return table_info

def save_kline_to_db(conn, cursor, prefix, code, name, df, start_date, end_date):
    """保存K线数据到数据库"""
    if df is None or df.empty:
        logger.warning(f"{code} 没有数据可保存")
        return None
    
    # 格式化日期为YYYYMMDD格式用于表名
    start_date_fmt = start_date.replace('-', '')
    end_date_fmt = end_date.replace('-', '')
    
    # 表名格式：line_code_startdate_enddate
    table_name = f"{prefix}_{code.replace('.', '')}_{name}_{start_date_fmt}_{end_date_fmt}_{len(df)}"
    
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    
    logger.info(f"已将 {code} 的 {len(df)} 条K线数据保存到表 {table_name}")
    return table_name

@with_db_connection
def get_all_stocks_latest_data(conn, cursor, freq='daily', length=30):
    """
    从数据库中读取所有股票的最后length行数据
    
    参数:
        freq: 数据频率，'daily' 或 'minute60'
        length: 要读取的最后几行数据
    
    返回:
        dict: {股票代码: DataFrame} 格式的字典
    """
    # 获取数据库中指定频率的所有表信息
    existing_tables = get_table_info_from_db(conn, cursor)
    freq_tables = {k: v for k, v in existing_tables.items() if v['freq'] == freq}
    
    if not freq_tables:
        logger.warning(f"数据库中没有找到 {freq} 频率的数据表")
        return {}
    
    logger.info(f"开始读取 {len(freq_tables)} 只股票的最后 {length} 行 {freq} 数据")
    
    stock_data = {}
    success_count = 0
    fail_count = 0
    
    for i, (code_clean, table_info) in enumerate(freq_tables.items()):
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
            
            # 按日期正序排列（因为我们用的是DESC，需要反转）
            df = df.sort_values('date').reset_index(drop=True)
            
            # 使用原始股票代码作为key（添加交易所前缀）
            original_code = restore_stock_code(code_clean)
            stock_data[original_code] = {
                'code': original_code,
                'name': stock_name,
                'data': df
            }
            
            logger.info(f"[{i+1}/{len(freq_tables)}] {original_code} ({stock_name}) 读取成功，获得 {len(df)} 行数据")
            success_count += 1
            
        except Exception as e:
            logger.error(f"[{i+1}/{len(freq_tables)}] {code_clean} 读取数据失败: {str(e)}")
            fail_count += 1
            continue
    
    logger.info(f"数据读取完成: 成功 {success_count}, 失败 {fail_count}")
    return stock_data

def restore_stock_code(code_clean):
    """
    将清理后的股票代码还原为原始格式
    
    参数:
        code_clean: 清理后的代码，如 'sh600000', 'sz000001'
    
    返回:
        str: 原始格式的股票代码，如 'sh.600000', 'sz.000001'
    """
    if code_clean.startswith('sh'):
        return f"{code_clean[2:]}.SH"
    elif code_clean.startswith('sz'):
        return f"{code_clean[2:]}.SZ"
    else:
        # 如果格式不符合预期，返回原值
        return code_clean

@with_db_connection
def get_specific_stocks_latest_data(conn, cursor, freq='daily', length=30):
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
        logger.warning(f"数据库中没有找到 {freq} 频率的数据表")
        return {}
    
    logger.info(f"开始读取 {len(freq_tables)} 只指定股票的最后 {length} 行 {freq} 数据")
    
    stock_data = {}
    success_count = 0
    fail_count = 0

    for i, (code, _) in enumerate(freq_tables.items()):
        # 将股票代码转换为数据库中的格式
        code_clean = code.replace('.', '')

        
        if code_clean not in freq_tables:
            logger.warning(f"[{i+1}/{len(freq_tables)}] {code_clean} 在数据库中未找到对应的 {freq} 数据表")
            fail_count += 1
            continue
        
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

@with_db_connection
@bs_login_required
def get_stock_industry(conn, cursor):
    """获取股票行业分类数据"""
    rs = bs.query_stock_industry()
    rs = rs.get_data()
    rs.to_sql('stock_industry', conn, if_exists='replace', index=False)
    return None

@with_db_connection
@bs_login_required
def update_stock_kline(conn, cursor, freq='daily', codes=None, force_update=False, process=False, max_stocks=None):
    """更新股票K线数据

    参数:
        codes: 要更新的股票代码列表，默认为None表示更新所有A股
        force_update: 是否强制更新所有数据，默认False只更新新数据
    """
    assert freq in ['daily', 'minute60'], "频率参数 freq 必须是 'daily' 或 'minute60'"
    # 获取最新的交易日作为today
    trade_dates = fetch_trade_dates()
    if trade_dates is None or trade_dates.empty:
        logger.error("无法获取交易日数据")
        return
    today = trade_dates.iloc[-1]
    
    # 获取数据库中已有的表信息
    existing_tables = get_table_info_from_db(conn, cursor)
    existing_tables = {k: v for k, v in existing_tables.items() if v['freq'] == freq}

    # 如果没有指定股票代码，则获取所有A股
    if codes is None:
        codes, names = fetch_stock_list()
        if codes is None:
            logger.error("无法获取股票列表，更新失败")
            return
    if max_stocks is not None:
        codes = codes[:max_stocks]
        names = names[:max_stocks]
    total_stocks = len(codes)
    logger.info(f"开始更新 {total_stocks} 只股票的K线数据")
    
    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, (code, name) in enumerate(zip(codes, names)):
        
        code_clean = code.replace('.', '')
            
        # 检查是否已存在该股票的数据表
        if code_clean in existing_tables.keys() and not force_update:
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
            if freq == 'daily':
                df = fetch_daily_kline(code, start_date=formatted_end_date, end_date=today)
            elif freq == 'minute60':
                df = fetch_minute60_kline(code, start_date=formatted_end_date, end_date=today)

            if df is not None and not df.empty and len(df) > 1:
                # 读取已有数据
                existing_df = pd.read_sql_query(f"SELECT * FROM {existing_info['table_name']}", conn)
                    
                # 合并新旧数据，去重
                merged_df = pd.concat([existing_df, df]).drop_duplicates(subset=['date'], keep='last')
                    
                # 获取合并后的数据范围
                new_start_date = merged_df['date'].min()
                new_end_date = merged_df['date'].max()
                    
                # 保存合并后的数据，并使用新的表名
                save_kline_to_db(conn, cursor, freq, code, name, merged_df, new_start_date, new_end_date)

                cursor.execute(f"DROP TABLE IF EXISTS {existing_info['table_name']}")
                    
                logger.info(f"[{i+1}/{total_stocks}] {code} 数据更新成功，从 {existing_end_date} 更新到 {new_end_date.replace('-', '')}")
                success_count += 1
            else:
                logger.warning(f"[{i+1}/{total_stocks}] {code} 无法获取新数据")
                fail_count += 1
        else:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
            df = fetch_daily_kline(code, start_date=start_date, end_date=today)
            if df is not None and not df.empty:
                # 保存数据到数据库
                save_kline_to_db(conn, cursor, freq, code, name, df, start_date, today)

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
    # update_stock_daily_kline(process=True,force_update=False)
    # print(DB_PATH)
    update_stock_kline(freq='daily')
    # get_stock_industry()
