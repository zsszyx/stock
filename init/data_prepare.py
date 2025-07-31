import os
import akshare as ak
import sys
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
# 动态定位当前工作目录中的 stock 目录
import efinance
import requests
import baostock as bs
import random
import adata as ad
import yaml

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

def load_proxies():
    """
    加载代理配置文件并返回代理列表。
    :return: 代理列表
    """
    proxy_file_path = "d:\\stock\\init\\proxy_pool.yaml"
    with open(proxy_file_path, "r", encoding="utf-8") as file:
        proxy_data = yaml.safe_load(file)
    return proxy_data.get("proxies", [])

def get_random_proxy(proxies):
    """
    从代理列表中随机选择一个代理并转换为 HTTP/HTTPS 格式。
    :param proxies: 代理列表
    :return: HTTP/HTTPS 格式的代理字典
    """
    import random
    selected_proxy = random.choice(proxies)
    proxy_url = f"http://{selected_proxy['server']}:{selected_proxy['port']}"
    return {
        "http": proxy_url,
        "https": proxy_url
    }

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
    today = (datetime.today() - timedelta(days=0)).strftime('%Y%m%d')  # 获取今天日期
    start_date = (datetime.today() - timedelta(days=n)).strftime('%Y%m%d')  # 获取 n 天前的日期
    return start_date, today


def fetch_history_stock_data_from_multiple_sources(symbol, period, start_date, end_date, adjust):
    """
    从多个接口随机获取股票数据，并计算振幅。
    随机选择 efinance、akshare、baostock 或 adata 进行数据获取，并支持代理。
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
    ak_to_ad_period = {'daily': 1, 'weekly': 2, 'monthly': 3}
    ak_to_ad_fqt = {
        'none': 0,
        'qfq': 1,
        'hfq': 2
    }
    adjustflag = str(ak_to_bao_period.get(adjust, 2))
    period_map = {'daily': 'd', 'weekly': 'w', 'monthly': 'm'}
    bao_frequency = period_map.get(period, 'd')

    # 定义接口列表
    interfaces = ["baostock"]

    # 随机选择一个接口
    selected_interface = random.choice(interfaces)

    # 定义代理列表（示例代理）
    proxy_list = [
        '27.189.133.114'
    ]

    # 随机选择一个代理
    selected_proxy = random.choice(proxy_list)

    if selected_interface == "efinance":
        try:
            klt = ak_to_ef_period.get(period, 101)
            fqt = ak_to_ef_fqt.get(adjust, 1)
            session = requests.Session()
            session.headers.update({"User-Agent": USER_AGENT})
            session.proxies.update({
                "http": f"http://{selected_proxy}",
                "https": f"http://{selected_proxy}"
            })
            efinance.stock.session = session
            df = efinance.stock.get_quote_history(
                symbol, beg=start_date, end=end_date, klt=klt, fqt=fqt
            )
            print(f"使用 efinance 获取数据 {symbol} {start_date}-{end_date}，代理：{selected_proxy}")
            return check_data(df)
        except Exception as e:
            print(f"efinance 获取失败: {e}")

    if selected_interface == "akshare":
        try:
            os.environ["HTTP_PROXY"] = f"http://{selected_proxy}"
            os.environ["HTTPS_PROXY"] = f"http://{selected_proxy}"
            df = ak.stock_zh_a_hist(
                symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust=adjust
            )
            print(f"使用 akshare 获取数据 {symbol} {start_date}-{end_date}，代理：{selected_proxy}")
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

            # 动态计算振幅
            df['振幅'] = ((df['最高'] - df['最低']) / df['收盘'].shift(1)) * 100

            print(f"使用 baostock 获取数据并计算振幅 {symbol} {start_date}-{end_date}，代理：{selected_proxy}")
            return check_data(df)
        except Exception as e:
            print(f"baostock 获取失败: {e}")

    if selected_interface == "adata":

        # 设置代理
        ad.proxy(is_proxy=True, ip=selected_proxy)

        k_type = ak_to_ad_period.get(period, 1)
        adjust_type = ak_to_ad_fqt.get(adjust, 1)
        df = ad.stock.market.get_market(
            stock_code=symbol, start_date=start_date, k_type=k_type, adjust_type=adjust_type
        )
        if df.empty:
            print(f"adata 获取数据失败，返回空表: {symbol} {start_date}-{end_date}")
            return pd.DataFrame()
        print(f"使用 adata 获取数据 {symbol} {start_date}-{end_date}，代理：{selected_proxy}")
        return check_data(df)

    # 如果所有接口都失败，返回空表
    print(f"无法获取数据 {symbol} {start_date}-{end_date}")
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
        if stock_zh_a_hist_df.empty:
            return pd.DataFrame()  # 返回空表
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

def get_stock_zh_a_hist_batch(params_list):
    """
    批量获取 A 股历史数据，优先使用数据库中已存在的表。
    如果数据库中不存在对应表，则调用 fetch_batch_kline_data_with_baostock 批量获取数据。
    保留原来的删除旧表操作。
    :param params_list: 参数列表，每个元素是一个字典，包含以下键：
        - symbol: 股票代码
        - period: 数据周期
        - start_date: 起始日期，格式为 'YYYYMMDD'
        - end_date: 结束日期，格式为 'YYYYMMDD'
        - adjust: 调整方式，可选值为 'qfq'、'hfq'、'none'
    :return: 一个列表，包含 (PARAM, DF) 元组
    """
    table_name_list = []
    existing_data = []
    fetch_list = []

    # 生成 table_name_list 并检查哪些表已存在
    cursor = conn.cursor()
    for params in params_list:
        symbol = params["symbol"]
        period = params["period"]
        start_date = params["start_date"]
        end_date = params["end_date"]
        adjust = params["adjust"]

        table_name = f"tbl_{symbol}_{period}_{adjust}_{start_date}_{end_date}"
        table_name_list.append(table_name)

        # 检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        exists = cursor.fetchone()
        if exists:
            # 如果表存在，直接读取数据
            stock_zh_a_hist_df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
            print(f"从数据库中读取表 {table_name}")
            existing_data.append((params, stock_zh_a_hist_df))
        else:
            # 如果表不存在，加入需要重新爬取的列表
            fetch_list.append(params)

    # 如果没有需要重新爬取的数据，直接返回
    if not fetch_list:
        return existing_data

    # 删除旧表
    for params in fetch_list:
        symbol = params["symbol"]
        period = params["period"]
        adjust = params["adjust"]
        prefix = f"tbl_{symbol}_{period}_{adjust}_"
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", (f"{prefix}%",))
        old_tables = cursor.fetchall()
        for old_table in old_tables:
            cursor.execute(f'DROP TABLE IF EXISTS "{old_table[0]}"')
            print(f"删除旧表: {old_table[0]}")

    # 批量爬取数据
    fetched_data = fetch_batch_kline_data_with_baostock(fetch_list)

    # 批量存储数据到数据库
    for params, df in fetched_data:
        symbol = params["symbol"]
        period = params["period"]
        start_date = params["start_date"]
        end_date = params["end_date"]
        adjust = params["adjust"]

        table_name = f"tbl_{symbol}_{period}_{adjust}_{start_date}_{end_date}"
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"获取数据并存储到表 {table_name}")

    # 合并已存在的数据和新爬取的数据
    return existing_data + fetched_data

def fetch_batch_kline_data_with_baostock(params_list):
    """
    使用 baostock 批量获取 K 线数据。
    :param params_list: 参数列表，每个元素是一个字典，包含以下键：
        - symbol: 股票代码
        - period: 数据周期
        - start_date: 起始日期，格式为 'YYYYMMDD'
        - end_date: 结束日期，格式为 'YYYYMMDD'
        - adjust: 调整方式，可选值为 'qfq'、'hfq'、'none'
    :return: 一个列表，包含 (PARAM, DF) 元组
    """
    fetched_data = []
    bs.login()  # 登录 baostock

    for params in params_list:
        symbol = params["symbol"]
        period = params["period"]
        start_date = params["start_date"]
        end_date = params["end_date"]
        adjust = params["adjust"]

        # 映射周期和复权方式
        period_map = {'daily': 'd', 'weekly': 'w', 'monthly': 'm'}
        bao_frequency = period_map.get(period, 'd')
        adjustflag = {'none': '3', 'qfq': '2', 'hfq': '1'}.get(adjust, '2')

        # 映射股票代码到 baostock 格式
        exchange = map_exchange_by_code(symbol)
        if exchange == 'sh':
            bao_code = f"sh.{symbol}"
        elif exchange == 'sz':
            bao_code = f"sz.{symbol}"
        else:
            print(f"跳过非沪深股票: {symbol}")
            continue

        # 获取 K 线数据
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

        # 转换为 DataFrame
        columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额', '换手率', '涨跌幅']
        df = pd.DataFrame(data_list, columns=columns)
        for col in ['开盘', '最高', '最低', '收盘', '成交量', '成交额', '换手率', '涨跌幅']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 动态计算振幅
        df['振幅'] = ((df['最高'] - df['最低']) / df['收盘'].shift(1)) * 100

        # 去掉第一行 NaN
        df = df.dropna(subset=['振幅','涨跌幅'])

        # 去掉满足条件的行（开盘=最高=收盘=最低，且成交量、成交额、换手率均为 NaN）
        df = df[~((df['开盘'] == df['最高']) & 
                  (df['开盘'] == df['收盘']) & 
                  (df['开盘'] == df['最低']) & 
                  df[['成交量', '成交额', '换手率']].isnull().all(axis=1))]

        print(f"使用 baostock 获取数据 {symbol} {start_date}-{end_date}")
        fetched_data.append((params, check_data(df)))

    bs.logout()  # 登出 baostock
    return fetched_data

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


if __name__ == "__main__":
    from validate import check_data
    fetch_history_stock_data_from_multiple_sources(
        symbol="000001", period="daily", start_date="20230801", end_date="20231231", adjust="qfq"
    )
else:
    from init import check_data