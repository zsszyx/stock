import os
import sqlite3
import pandas as pd
import baostock as bs
import datetime
import logging
import time
import contextlib
import functools

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_data.db')

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

@bs_login_required
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
    logger.info(f"获取到 {len(df)} 只股票信息")
    return df['code']

if __name__ == "__main__":
    # fetch_stock_list()
    print(fetch_trade_dates())