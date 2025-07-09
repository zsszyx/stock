import os
import akshare as ak
import sys
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
# 动态定位当前工作目录中的 stock 目录
from init import check_data
import efinance
import requests
import baostock as bs
import random

# 全局反扒 user-agent 设置
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"

# 设置 requests 的全局 user-agent
requests_session = requests.Session()
requests_session.headers.update({
    "User-Agent": USER_AGENT
})
# efinance 支持 session
efinance.stock.session = requests_session

# akshare 通过环境变量设置 user-agent
os.environ["AKSHARE_REQUEST_HEADERS"] = f'{{"User-Agent": "{USER_AGENT}"}}'

# 全局持久连接
DB_PATH = 'stock_data.db'
conn = sqlite3.connect(DB_PATH)  # 创建全局数据库连接

def initialize_data():
    """
    初始化股票数据，获取 A 股现货数据并划分为不同板块，保存到 SQLite 数据库。
    """
    # 设置 Pandas 显示选项，确保打印时不截断
    pd.set_option('display.max_rows', None)  # 显示所有行

    stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()

    # 过滤掉“名称”列包含 ST 或 PT 以及退市的股票
    filtered_df = stock_zh_a_spot_em_df[~stock_zh_a_spot_em_df['名称'].str.contains('ST|PT|退|转', na=False)]

    # 去掉代码列以 900、689 和 200 开头的行
    filtered_df = filtered_df[~filtered_df['代码'].str.startswith(('900', '689', '200'))]

    # 划分不同板块
    main_board_df = filtered_df[filtered_df['代码'].str.startswith(('60', '00'))]  # 主板
    science_board_df = filtered_df[filtered_df['代码'].str.startswith('688')]      # 科创板
    growth_board_df = filtered_df[filtered_df['代码'].str.startswith('30')]        # 创业板
    beijing_board_df = filtered_df[filtered_df['代码'].str.startswith(('83', '87', '430', '92'))]  # 北交所

    # 检查未划分的行
    remaining_df = filtered_df[
        ~filtered_df['代码'].str.startswith(('60', '00', '688', '30', '83', '87', '430', '92'))
    ]

    # 使用 assert 确保未划分的表没有行
    assert remaining_df.empty, "存在未划分的股票，请检查过滤条件！"

    # 将每个 DataFrame 保存为单独的表
    main_board_df.to_sql('main_board', conn, if_exists='replace', index=False)
    science_board_df.to_sql('science_board', conn, if_exists='replace', index=False)
    growth_board_df.to_sql('growth_board', conn, if_exists='replace', index=False)
    beijing_board_df.to_sql('beijing_board', conn, if_exists='replace', index=False)

    print("数据已保存到 SQLite 数据库 'stock_data.db'")


def get_trade_date():
    tool_trade_date_hist_sina_df = ak.tool_trade_date_hist_sina()
    return tool_trade_date_hist_sina_df


def get_start_to_end_date(n=180):
    '''
    获取最近 n 天的交易日期范围
    :param n: 最近天数，默认为 180 天
    :return: 最近 n 天的交易日期范围，格式为 'YYYY-MM-DD'
    '''
    today = datetime.today().strftime('%Y%m%d')  # 获取今天日期
    start_date = (datetime.today() - timedelta(days=n)).strftime('%Y%m%d')  # 获取 n 天前的日期
    return start_date, today


def fetch_history_stock_data_from_multiple_sources(symbol, period, start_date, end_date, adjust):
    """
    从多个接口随机获取股票数据。
    随机选择 efinance、akshare 或 baostock 进行数据获取。
    :param symbol: 股票代码
    :param period: 数据周期
    :param start_date: 起始日期
    :param end_date: 结束日期
    :param adjust: 调整方式
    :return: 股票数据 DataFrame
    """
    # 映射字典
    ak_to_ef_period = {
        'daily': 101,
        'weekly': 102,
        'monthly': 103
    }
    ak_to_ef_fqt = {
        'none': 0,
        'qfq': 1,
        'hfq': 2
    }
    ak_to_bao_period = {
        'hfq': 1,
        'qfq': 2,
        'none': 3
    }
    adjustflag = str(ak_to_bao_period.get(adjust, 2))
    period_map = {'daily': 'd', 'weekly': 'w', 'monthly': 'm'}
    bao_frequency = period_map.get(period, 'd')

    # 定义接口列表
    interfaces = ["efinance", "akshare", "baostock"]

    # 随机选择一个接口
    selected_interface = random.choice(interfaces)

    if selected_interface == "efinance":
        try:
            klt = ak_to_ef_period.get(period, 101)
            fqt = ak_to_ef_fqt.get(adjust, 1)
            df = efinance.stock.get_quote_history(
                symbol, beg=start_date, end=end_date, klt=klt, fqt=fqt
            )
            print(f"使用 efinance 获取数据 {symbol} {start_date}-{end_date}")
            return check_data(df)
        except Exception as e:
            print(f"efinance 获取失败: {e}")

    if selected_interface == "akshare":
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust=adjust
            )
            print(f"使用 akshare 获取数据 {symbol} {start_date}-{end_date}")
            return check_data(df)
        except Exception as e:
            print(f"akshare 获取失败: {e}")

    if selected_interface == "baostock":
        try:
            exchange = map_exchange_by_code(symbol)
            if exchange == 'sh':
                bao_code = f"sh.{symbol}"
            elif exchange == 'sz':
                bao_code = f"sz.{symbol}"
            else:
                print(f"跳过非沪深股票: {symbol}")
                return pd.DataFrame()  # 返回空表
            bs.login()
            rs = bs.query_history_k_data_plus(
                bao_code,
                "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start_date[:4] + '-' + start_date[4:6] + '-' + start_date[6:],
                end_date=end_date[:4] + '-' + end_date[4:6] + '-' + end_date[6:],
                frequency=bao_frequency,
                adjustflag=adjustflag
            )
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            bs.logout()
            columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额', '换手率', '涨跌幅']
            df = pd.DataFrame(data_list, columns=columns)
            for col in ['开盘', '最高', '最低', '收盘', '成交量', '成交额', '换手率', '涨跌幅']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"使用 baostock 获取数据 {symbol} {start_date}-{end_date}")
            return check_data(df)
        except Exception as e:
            print(f"baostock 获取失败: {e}")

    # 如果所有接口都失败，返回空表
    print(f"所有接口均失败，无法获取数据 {symbol} {start_date}-{end_date}")
    return pd.DataFrame()


def get_stock_zh_a_hist(symbol="000001", period="daily", start_date="20180301", end_date="20240528", adjust="qfq"):
    """
    获取 A 股历史数据，优先使用 efinance，避免重复调用接口。
    如果数据库中存在对应表，则直接读取；否则调用 fetch_history_stock_data_from_multiple_sources 获取数据并存储。
    :param symbol: 股票代码，默认为 000001
    :param period: 数据周期，默认为 daily
    :param start_date: 起始日期，格式为 'YYYYMMDD'
    :param end_date: 结束日期，格式为 'YYYYMMDD'
    :param adjust: 调整方式，默认为前复权（qfq），可选值为 'qfq'、'hfq'、'none'
    :return: A 股历史数据 DataFrame
    """
    table_name = f"tbl_{symbol}_{period}_{adjust}_{start_date}_{end_date}"

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    exists = cursor.fetchone()

    if exists:
        stock_zh_a_hist_df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
        print(f"从数据库中读取表 {table_name}")
    else:
        # 调用新方法获取数据
        stock_zh_a_hist_df = fetch_history_stock_data_from_multiple_sources(symbol, period, start_date, end_date, adjust)

        # 删除以 {symbol}_{period}_{adjust} 开头的旧表
        prefix = f"tbl_{symbol}_{period}_{adjust}_"
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", (f"{prefix}%",))
        old_tables = cursor.fetchall()
        for old_table in old_tables:
            cursor.execute(f'DROP TABLE IF EXISTS "{old_table[0]}"')
            print(f"删除旧表: {old_table[0]}")

        # 存储到数据库
        stock_zh_a_hist_df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"获取数据并存储到表 {table_name}")

    return stock_zh_a_hist_df

def read_main_board():
    """
    读取主板表中的代码、名称和总市值，并返回一个元组列表。
    :return: 主板表中的代码、名称和总市值组成的元组列表 [(代码, 名称, 总市值), ...]
    """
    cursor = conn.cursor()

    # 检查主板表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='main_board'")
    exists = cursor.fetchone()
    if not exists:
        raise ValueError("主板表不存在，请确保数据库已正确初始化！")

    # 读取主板表中的代码、名称和总市值
    cursor.execute("SELECT 代码, 名称, 总市值 FROM main_board")
    rows = cursor.fetchall()
    
    # 返回元组列表
    return rows

def read_growth_board():
    """
    读取创业板表中的代码、名称和总市值，并返回一个元组列表。
    :return: 创业板表中的代码、名称和总市值组成的元组列表 [(代码, 名称, 总市值), ...]
    """
    cursor = conn.cursor()

    # 检查创业板表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='growth_board'")
    exists = cursor.fetchone()
    if not exists:
        raise ValueError("创业板表不存在，请确保数据库已正确初始化！")

    # 读取创业板表中的代码、名称和总市值
    cursor.execute("SELECT 代码, 名称, 总市值 FROM growth_board")
    rows = cursor.fetchall()
    
    # 返回元组列表
    return rows

def map_exchange_by_code(stock_code):
    """
    根据股票代码映射交易所。
    :param stock_code: 股票代码
    :return: 交易所代码 ('sh', 'sz', 'bj', 'hk')
    """
    if stock_code.startswith('6'):
        return 'sh'
    elif stock_code.startswith(('0', '3')):
        return 'sz'
    elif stock_code.startswith(('9', '8', '4')):
        return 'bj'
    elif len(stock_code) == 5:
        return 'hk'
    else:
        raise ValueError(f"无法映射交易所，未知代码格式: {stock_code}")


