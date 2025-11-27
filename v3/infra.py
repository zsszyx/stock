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

"""
重构后的infra.py，实现股票数据存储在一个大表中并支持个股增量更新

数据表结构：
- daily_stock_data: 存储所有股票的日线数据
- minute5_stock_data: 存储所有股票的5分钟线数据
- stock_industry: 存储股票行业分类数据
- stock_info: 存储股票基本信息
"""

# ==================== 全局配置 ====================

# 数据库配置
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stock_data.db')

# 日志配置
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# 数据表配置
TABLE_NAMES = {
    'daily': 'daily_stock_data',
    'minute5': 'minute5_stock_data',
    'industry': 'stock_industry',
    'info': 'stock_info'
}

# 数据表创建SQL语句
CREATE_SQL_STATEMENTS = {
    'daily': '''
    CREATE TABLE IF NOT EXISTS daily_stock_data (
        code TEXT,
        name TEXT,
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        turn REAL,
        amount REAL,
        pctChg REAL,
        peTTM REAL,
        pbMRQ REAL,
        psTTM REAL,
        pcfNcfTTM REAL,
        market_value REAL,
        PRIMARY KEY (code, date)
    )
    ''',
    'minute5': '''
    CREATE TABLE IF NOT EXISTS minute5_stock_data (
        code TEXT,
        name TEXT,
        date TEXT,
        time TEXT,
        volume REAL,
        amount REAL,
        PRIMARY KEY (code, time)
    )
    '''
}

# K线数据字段配置
KLINE_FIELDS = {
    'daily': 'date,open,high,low,close,volume,turn,amount,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM',
    'minute5': 'date,time,volume,amount'
}

# 复权类型配置（2-前复权，1-后复权，3-不复权）
DEFAULT_ADJUST_FLAG = {
    'daily': '2',
    'minute5': '3'
}

# 首次/强制更新默认时间范围（两年）
FIRST_UPDATE_DAYS = 365

# 更新配置 # 是否强制更新（无视上次更新时间）
DEFAULT_FORCE_UPDATE = DEFAULT_FORCE_RECREATE = False  # 是否强制重建表结构

# 更新到最近哪一天
LATEST_DAY = -1  # 默认更新到最近一天

# 初始化日志
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

print(f"数据库路径: {DB_PATH}")

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
def fetch_trade_dates(start_date=None, end_date=None):
    """获取交易日数据"""
    logger.info(f"获取交易日数据: {start_date} 至 {end_date}")
    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    df = rs.get_data()
    if df.empty:
        raise ValueError("获取的交易日数据为空")
    # 筛选出是交易日的记录并按日期降序排序
    df = df[df['is_trading_day'] == '1']
    logger.info(f"获取到 {len(df)} 条交易日数据")
    return df['calendar_date']

# ================= 股票基本信息 =================
def fetch_stock_list(start_date=None, end_date=None):
    """获取股票列表"""
    logger.info(f"获取 {end_date} 的股票列表")
    rs = bs.query_all_stock(end_date)
    df = rs.get_data()
    index_df = df[df['code'].str.startswith(('sh.000', 'sz.399'))]
    df = df[~df['code'].str.startswith(('sh.000', 'sz.399'))]
    df = df[~df['code_name'].str.contains(r'\*|ST|S|退')]
    logger.info(f"获取到 {len(df)} 只股票信息")
    if df['code'].isnull().any() or df.empty:
        raise ValueError("股票代码中存在空值, 或者获取的股票列表为空")
    # 增加一行上证指数
    df = pd.concat([pd.DataFrame([{'code': 'sh.000001', 'code_name': '上证指数'}]), df], ignore_index=True)
    return df['code'], df['code_name'], index_df['code'], index_df['code_name']

# ================= K线数据获取 =================
def fetch_daily_kline(code, start_date=None, end_date=None, adjustflag=None):
    """获取股票日K线前复权数据
    
    参数:
        code: 股票代码，格式如 sh.600000
        start_date: 开始日期，默认为一年前
        end_date: 结束日期，默认为昨天
        adjustflag: 复权类型，2-前复权，1-后复权，3-不复权，默认从全局配置读取
    """
    # 使用全局配置的复权类型
    if adjustflag is None:
        adjustflag = DEFAULT_ADJUST_FLAG['daily']
    
    logger.info(f"获取 {code} 从 {start_date} 到 {end_date} 的日K线数据")
    
    # 使用全局配置的K线数据字段
    fields = KLINE_FIELDS['daily']

    rs = bs.query_history_k_data_plus(code, fields,
                                      start_date=start_date,
                                      end_date=end_date,
                                      frequency="d", 
                                      adjustflag=adjustflag)
    
    # 手动构建DataFrame，避免使用已弃用的append方法
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    
    # 如果没有数据，返回空的DataFrame
    if not data_list:
        logger.info("未获取到日K线数据")
        return pd.DataFrame(columns=fields.split(','))
    
    # 创建DataFrame
    df = pd.DataFrame(data_list, columns=fields.split(','))

    if rs.error_code != '0' or df.empty:
        logger.error(f"获取daily K线数据失败: {rs.error_msg}")
        return None
    
    # 将能转换成数值的列都转换成数值类型
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col])

    df['turn'] = df['turn'].replace(0, np.nan)
    df['market_value'] = df['close'] * df['volume'] / df['turn']
    df['market_value'] = df['market_value'].ffill()
    df['market_value'] = df['market_value'].bfill()
    logger.info(f"获取到 {len(df)} 条K线数据")
    return df

def fetch_minute5_kline(code, start_date=None, end_date=None, adjustflag=None):
    """获取股票5分钟K线数据

    参数:
        code: 股票代码，格式如 sh.600000
        start_date: 开始日期，默认为一年前
        end_date: 结束日期，默认为昨天
        adjustflag: 复权类型，2-前复权，1-后复权，3-不复权，默认从全局配置读取
    """
    # 使用全局配置的复权类型
    if adjustflag is None:
        adjustflag = DEFAULT_ADJUST_FLAG['minute5']
    
    logger.info(f"获取 {code} 从 {start_date} 到 {end_date} 的分钟K线数据")

    # 使用全局配置的K线数据字段
    fields = KLINE_FIELDS['minute5']

    rs = bs.query_history_k_data_plus(code, fields,
                                      start_date=start_date,
                                      end_date=end_date,
                                      frequency="5",
                                      adjustflag=adjustflag)

    if rs.error_code != '0':
        logger.error(f"获取minute K线数据失败: {rs.error_msg}")
        return None
    
    # 手动构建DataFrame，避免使用已弃用的append方法
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    
    # 如果没有数据，返回空的DataFrame
    if not data_list:
        logger.info("未获取到分钟K线数据")
        return pd.DataFrame(columns=fields.split(','))
    
    # 创建DataFrame
    df = pd.DataFrame(data_list, columns=fields.split(','))

    # 将能转换成数值的列都转换成数值类型
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col])

    logger.info(f"获取到 {len(df)} 条分钟K线数据")
    return df

# ================= 数据库操作相关 =================
def ensure_table_exists(conn, cursor, table_name, create_sql, force_recreate=None):
    """确保数据表存在，不存在则创建
    
    参数:
        force_recreate: 是否强制删除并重新创建表，默认为None（使用全局配置）
    """
    # 使用全局配置或传入的参数
    force_recreate = DEFAULT_FORCE_RECREATE if force_recreate is None else force_recreate
    
    # 检查是否强制重建表
    if force_recreate:
        logger.info(f"强制删除并重建表: {table_name}")
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(create_sql)
        conn.commit()
        return True
    
    # 检查是否需要创建新表
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    if not cursor.fetchone():
        logger.info(f"创建表: {table_name}")
        cursor.execute(create_sql)
        conn.commit()
    
    return True

def get_stock_last_update_date(conn, cursor, table_name, code):
    """获取指定股票在指定表中的最后更新日期"""
    try:
        # 先检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
             logging.info(f"表 {table_name} 不存在，返回None")
             return None
        
        # 检查表是否是分钟线表，如果是则同时考虑date和time列
        if table_name == TABLE_NAMES['minute5']:
            # 对于分钟线表，获取最新的记录（同时考虑date和time）
            query = f"""SELECT date FROM {table_name} WHERE code = ? ORDER BY date DESC, time DESC LIMIT 1"""
        else:
            # 对于其他表，使用MAX(date)查询
            query = f"""SELECT MAX(date) FROM {table_name} WHERE code = ?"""
        
        cursor.execute(query, (code,))
        result = cursor.fetchone()
        # 处理None返回值
        return result[0] if result and result[0] is not None else None
    except Exception as e:
         logging.error(f"获取股票 {code} 的最后更新日期时出错: {e}")
         return None

@with_db_connection
def get_industry_data(conn, cursor):
    """获取股票行业分类数据"""
    sql = "SELECT * FROM stock_industry"
    df = pd.read_sql(sql, conn)
    # 把code从sh.000001 转换为 sh000001
    df['code'] = df['code'].str.replace('sh.', 'sh')
    # 把code从sz.000001 转换为 sz000001
    df['code'] = df['code'].str.replace('sz.', 'sz')
    return df[['code', 'industry']]

def update_industry(conn, cursor, today):
    """获取股票行业分类数据"""
    rs = bs.query_stock_industry(date=today)
    rs = rs.get_data()
    if rs.empty:
        raise ValueError("获取的行业分类数据为空")
    rs.to_sql('stock_industry', conn, if_exists='replace', index=False)
    return rs

# ================= 数据更新核心函数 =================
def update_single_stock_data(conn, cursor, code, name, freq, today, force_update=None):
    """更新单个股票的数据到大表中
    
    注意：表结构相关操作已移至update_stock_kline函数中全局处理
    """
    # 使用全局配置或传入的参数
    force_update = DEFAULT_FORCE_UPDATE if force_update is None else force_update
    # 验证频率参数
    if freq not in TABLE_NAMES:
        raise ValueError(f"不支持的频率: {freq}")
    
    # 从全局配置获取表名和创建SQL
    table_name = TABLE_NAMES[freq]
    fetch_func = fetch_daily_kline if freq == 'daily' else fetch_minute5_kline
    
    # 获取该股票的最后更新日期
    last_update_date = get_stock_last_update_date(conn, cursor, table_name, code)
    
    # 确定要获取的数据范围
    if force_update or last_update_date is None:
        # 强制更新或首次更新，使用全局配置的天数
        start_date = (datetime.datetime.now() - datetime.timedelta(days=FIRST_UPDATE_DAYS)).strftime('%Y-%m-%d')
        logging.info(f"[{code}] 首次更新或强制更新，获取 {start_date} 至 {today} 的数据")
    else:
        # 增量更新，从上一次更新日期的下一天开始
        # 将last_update_date转换为datetime对象并加一天
        last_date_obj = datetime.datetime.strptime(last_update_date, '%Y-%m-%d')
        next_day_obj = last_date_obj + datetime.timedelta(days=1)
        start_date = next_day_obj.strftime('%Y-%m-%d')
        logging.info(f"[{code}] 增量更新，获取 {start_date} 至 {today} 的数据")
    
    # 获取数据
    df = fetch_func(code, start_date=start_date, end_date=today)
    
    if df is None or df.empty:
        logger.warning(f"[{code}] 无法获取数据")
        return False
    
    # 添加股票代码和名称
    df['code'] = code
    df['name'] = name
    
    # 数据去重 - 对于分钟线数据，使用(code, time)组合确保唯一性
    initial_len = len(df)
    if 'time' in df.columns:
        df = df.drop_duplicates(subset=['code', 'time'], keep='last')
    else:
        df = df.drop_duplicates(subset=['code', 'date'], keep='last')
    if len(df) < initial_len:
        logging.warning(f"[{code}] 发现 {initial_len - len(df)} 条重复数据，已去重")
    
    # 对于增量更新，先删除已有的相应日期的数据
    # if not force_update and last_update_date is not None and len(df) > 0:
    #     # 删除从start_date开始的所有数据（现在是last_update_date的下一天）
    #     # 对于分钟线表，虽然主键已变更为(code, time)，但我们仍基于date进行增量更新
    #     delete_query = f"DELETE FROM {table_name} WHERE code = ? AND date >= ?"
    #     cursor.execute(delete_query, (code, start_date))
    #     conn.commit()  # 确保删除操作立即生效
    #     logging.info(f"[{code}] 已删除 {start_date} 之后的旧数据，为增量更新做准备")
    
    # 插入数据
    try:
        if len(df) > 0:
            # 构建INSERT OR REPLACE语句
            columns_str = ','.join(df.columns)
            placeholders = ','.join(['?'] * len(df.columns))
            sql = f"INSERT OR REPLACE INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            
            # 准备数据元组列表
            data_tuples = [tuple(row) for _, row in df.iterrows()]
            
            # 执行批量插入
            cursor.executemany(sql, data_tuples)
            conn.commit()
            
            logging.info(f"[{code}] 成功批量插入/更新 {len(df)} 条数据到 {table_name}")
            return True
        else:
            logging.info(f"[{code}] 没有新数据需要插入")
            return True
    except Exception as e:
        conn.rollback()  # 发生错误时回滚事务
        logging.error(f"[{code}] 数据处理失败: {str(e)}")
        return False

@with_db_connection
@bs_login_required
def update_stock_kline(conn, cursor, freq='daily'):
    """更新股票K线数据到大表中，支持个股增量更新

    参数:
        freq: 数据频率，'daily' 或 'minute5'
        codes: 要更新的股票代码列表，默认为None表示更新所有A股
        force_update: 是否强制更新所有数据，默认为None（使用全局配置）
        force_recreate: 是否强制重建表结构，默认为None（使用全局配置）
        max_stocks: 最大更新股票数量，用于测试
    """
    # 使用全局配置或传入的参数
    force_update = DEFAULT_FORCE_UPDATE 
    force_recreate = DEFAULT_FORCE_RECREATE 
    assert freq in ['daily', 'minute5'], "频率参数 freq 必须是 'daily' 或 'minute5'"
    
    # 从全局配置获取表名和创建SQL
    table_name = TABLE_NAMES[freq]
    create_sql = CREATE_SQL_STATEMENTS[freq]
    
    # 全局层面执行表检查和重建操作（每个更新批次只执行一次）
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        # 表不存在时创建表
        ensure_table_exists(conn, cursor, table_name, create_sql, force_recreate=False)
        logging.info(f"全局: 表 {table_name} 不存在，已创建")
    elif force_recreate:
        # 表存在且用户要求重建时才重建（全局只重建一次）
        ensure_table_exists(conn, cursor, table_name, create_sql, force_recreate=True)
        logging.info(f"全局: 表 {table_name} 已强制重建")
    
    # 获取最新的交易日作为today
    trade_dates = fetch_trade_dates()
    today = trade_dates.iloc[LATEST_DAY]
    logging.info(f"最新交易日: {today}")
    
    # 获取股票列表
    all_codes, all_names, index_codes, index_names = fetch_stock_list(end_date=today)
    
    codes, names = all_codes, all_names
    
    total_stocks = len(codes)
    logger.info(f"开始更新 {total_stocks} 只股票的{freq}数据到大表中")
    
    success_count = 0
    fail_count = 0
    
    for i, (code, name) in enumerate(zip(codes, names)):
        logging.info(f"[{i+1}/{total_stocks}] 处理股票: {code} - {name}")
        # 更新单个股票数据
        if update_single_stock_data(conn, cursor, code, name, freq, today, force_update):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"K线数据更新完成: 成功 {success_count}, 失败 {fail_count}")

# ================= 数据查询函数 =================
@with_db_connection
def get_stock_merge_table(conn, cursor, length=None, freq='daily', start_date=None, end_date=None):
    """
    获取所有股票的最新数据，拼接为一个大表
    计算成交均价并按股票代码分组保留最近数据
    对于日线数据，使用日期范围查询保留最近20个交易日的数据
    对于分钟线数据，保留最近960条记录
    """
    # 验证频率参数
    if freq not in TABLE_NAMES:
        raise ValueError(f"不支持的频率: {freq}")
    
    # 从全局配置获取表名
    table_name = TABLE_NAMES[freq]
    
      
    # 构建数据查询 - 只获取从start_date开始的数据
    query = f"""
        SELECT * FROM {table_name}
        WHERE date >= ? AND date <= ?
        ORDER BY code, date
        """
    # 执行查询
    logging.info(f"开始读取所有股票从 {start_date} 至 {end_date} 的日线数据")
    df = pd.read_sql(query, conn, params=[start_date, end_date])
                
    df['price'] = df['amount'] / df['volume'].replace(0, pd.NA)
    
    # 设置索引
    merged = df.set_index(['code', 'date'])
    
    logging.info(f"成功读取并处理 {len(df['code'].unique())} 只股票的数据")
    return merged

if __name__ == "__main__":
    # 示例用法 - 使用全局配置
    # 更新日线数据
    # update_stock_kline(freq='daily')  # 使用默认配置（无需强制更新和重建）
    
    # 获取合并行业信息的股票数据（直接SQL查询，无需中转函数）
    # 注意：现在函数已添加@with_db_connection装饰器，不需要手动传入连接参数
    # stock_industry_df = get_stock_merge_industry_table(length=30, freq='daily')
    
    # 如果需要临时修改默认行为，可以直接修改全局配置
    # 然后调用函数时不传入相关参数
    # 例如：
    # global DEFAULT_FORCE_UPDATE, DEFAULT_FORCE_RECREATE
    # DEFAULT_FORCE_UPDATE = True
    # DEFAULT_FORCE_RECREATE = True
    # update_stock_kline(freq='minute5')  # 现在会使用修改后的全局配置
    # 
    # 或者直接在调用时传入参数（会覆盖全局配置）
    # update_stock_kline(freq='minute5', latest_day=-2)
    
    # 获取合并表数据（使用全局配置的默认长度）
    df = get_stock_merge_table(freq='minute5', start_date='2025-11-12', end_date='2025-11-26')
    df.to_csv('stock_data.csv')
    print(df.head(1000))
    print(df.info())