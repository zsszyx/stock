import pandas as pd
import warnings
from init import *
from init.data_prepare import read_main_board, read_growth_board, get_stock_zh_a_hist, get_start_to_end_date

def mark_long_erbo_condition1(df: pd.DataFrame):
    """
    长二波条件1：
    对 DataFrame 按照 15 行的滑动窗口判断：
    - 过去五天涨跌幅小于 5%
    - 前十天有涨跌幅大于 9%
    在 DataFrame 中增加一列 '长二波条件1'，标记当天是否符合条件。
    :param df: 包含股票数据的 DataFrame，必须包含 '涨跌幅' 列
    :return: 增加标记列的 DataFrame
    """
    if '涨跌幅' not in df.columns:
        raise ValueError("DataFrame 必须包含 '涨跌幅' 列！")

    # 初始化标记列
    df['长二波条件1'] = False

    # 滑动窗口处理
    for i in range(len(df) - 14):  # 确保窗口范围在数据长度内
        window = df.iloc[i:i + 15]  # 取 15 行的滑动窗口
        past_5_days = window.iloc[-5:]  # 过去五天
        past_10_days = window.iloc[:-5]  # 前十天

        # 判断条件
        if past_5_days['涨跌幅'].max() < 5 and past_10_days['涨跌幅'].max() > 9:
            df.at[i + 14, '长二波条件1'] = True  # 标记当天符合条件

    return df

def mark_short_erbo_condition1(df: pd.DataFrame):
    """
    短二波条件1：
    对 DataFrame 按照 10 行的滑动窗口判断：
    - 前 7 天有涨跌幅大于 9%
    - 后 3 天涨跌幅小于 5%
    在 DataFrame 中增加一列 '短二波条件1'，标记当天是否符合条件。
    :param df: 包含股票数据的 DataFrame，必须包含 '涨跌幅' 列
    :return: 增加标记列的 DataFrame
    """
    if '涨跌幅' not in df.columns:
        raise ValueError("DataFrame 必须包含 '涨跌幅' 列！")

    # 初始化标记列
    df['短二波条件1'] = False

    # 滑动窗口处理
    for i in range(len(df) - 9):  # 确保窗口范围在数据长度内
        window = df.iloc[i:i + 10]  # 取 10 行的滑动窗口
        past_7_days = window.iloc[:7]  # 前 7 天
        last_3_days = window.iloc[7:]  # 后 3 天

        # 判断条件
        if past_7_days['涨跌幅'].max() > 9 and last_3_days['涨跌幅'].max() < 5:
            df.at[i + 9, '短二波条件1'] = True  # 标记当天符合条件

    return df

def filter_zhu_erbo_condition0():
    """
    朱二波条件0：
    使用读取主板和创业板的方法拿到元组列表，并筛选总市值小于 100 亿的元素。
    对总市值为 None 的元素抛出 Warning，并显示对应的代码和名称。
    :return: 筛选后的元组列表 [(代码, 名称, 总市值), ...]
    """
    # 调用读取主板和创业板的方法
    main_board_data = read_main_board()
    growth_board_data = read_growth_board()

    # 合并主板和创业板数据
    all_board_data = main_board_data + growth_board_data

    # 筛选总市值小于 100 亿的元素，并对 None 值抛出 Warning
    filtered_data = []
    for row in all_board_data:
        if row[2] is None:
            warnings.warn(f"总市值为 None 的元素: 代码={row[0]}, 名称={row[1]}")
        elif int(row[2]) < 100_000_000_000:
            filtered_data.append(row)

    return filtered_data

def mark_zhu_erbo_condition1(df: pd.DataFrame):
    """
    朱二波条件1：
    标记最近十天内有涨跌幅大于 9.85%，并且当天满足以下条件：
    - 涨跌幅在 -1% 到 3% 之间。
    - 振幅小于 5%。
    - 换手率大于 2%。
    在 DataFrame 中增加一列 '朱二波条件1'，标记当天是否符合条件。
    :param df: 包含股票数据的 DataFrame，必须包含 '涨跌幅'、'振幅' 和 '换手率' 列
    :return: 增加标记列的 DataFrame
    """
    required_columns = ['涨跌幅', '振幅', '换手率']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    # 初始化标记列
    df['朱二波条件1'] = False

    # 滑动窗口处理
    for i in range(len(df) - 9):  # 确保窗口范围在数据长度内
        window = df.iloc[i:i + 10]  # 取最近十天的滑动窗口

        # 判断条件
        if window['涨跌幅'].max() > 9.85 and -1 <= window.iloc[-1]['涨跌幅'] <= 3 and window.iloc[-1]['振幅'] < 5 and window.iloc[-1]['换手率'] > 2:
            df.at[i + 9, '朱二波条件1'] = True  # 标记当天符合条件

    return df

def mark_consecutive_small_positive(df: pd.DataFrame):
    """
    寻找连续的缩量小阳线：
    - 最近 20 天成交量的平均值和标准差。
    - 成交量小于均值减标准差且涨跌幅大于 0 不超过 3。
    - 统计符合条件的连续天数，并记录在一列 '连续缩量小阳线' 中。
    :param df: 包含股票数据的 DataFrame，必须包含 '成交量' 和 '涨跌幅' 列。
    :return: 增加标记列的 DataFrame。
    """
    required_columns = ['成交量', '涨跌幅']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    # 初始化标记列
    df['连续缩量小阳线'] = 0

    # 滑动窗口处理
    consecutive_days = 0  # 记录连续天数
    for i in range(len(df)):
        # 计算最近 20 天的均值和标准差
        window = df.iloc[max(0, i - 19):i + 1]  # 最近 20 天的窗口
        volume_mean = window['成交量'].mean()
        volume_std = window['成交量'].std()

        # 判断当天是否符合条件
        if df.at[i, '成交量'] < (volume_mean - volume_std) and 0 < df.at[i, '涨跌幅'] <= 3:
            consecutive_days += 1
            df.at[i, '连续缩量小阳线'] = consecutive_days
        else:
            consecutive_days = 0  # 不符合条件时重置连续天数

    return df

def erbo_main_query_mode():
    """
    二波主函数-查询模式：
    1. 过滤股票名单
    2. 查询最近30天历史行情
    3. 应用所有mark方法
    4. 统计最新一天命中条件数量，排序打印
    """
    # 1. 过滤股票名单
    filtered_stocks = filter_zhu_erbo_condition0()
    if not filtered_stocks:
        print("没有符合条件的股票。")
        return

    results = []
    for code, name, _ in filtered_stocks:
        # 2. 查询最近30天历史行情
        start_date, end_date = get_start_to_end_date(30)
        try:
            df = get_stock_zh_a_hist(symbol=code, start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"{code} {name} 获取行情失败: {e}")
            continue

        # 3. 应用所有mark方法
        try:
            df = mark_long_erbo_condition1(df)
            df = mark_short_erbo_condition1(df)
            df = mark_zhu_erbo_condition1(df)
            df = mark_consecutive_small_positive(df)
        except Exception as e:
            print(f"{code} {name} 计算条件失败: {e}")
            continue

        # 4. 统计最新一天命中条件数量
        last = df.iloc[-1]
        hit_count = int(last.get('长二波条件1', False)) + \
                    int(last.get('短二波条件1', False)) + \
                    int(last.get('朱二波条件1', False))
        small_positive = int(last.get('连续缩量小阳线', 0))
        if hit_count > 0 and small_positive > 0:
            results.append({
                "code": code,
                "name": name,
                "hit_count": hit_count,
                "small_positive": small_positive
            })

    # 5. 排序并打印
    results.sort(key=lambda x: (-x['hit_count'], -x['small_positive']))
    print("命中条件数量排序结果：")
    for r in results:
        print(f"{r['name']}({r['code']}): 命中{r['hit_count']}个条件, 连续缩量小阳线{r['small_positive']}天")

if __name__ == "__main__":
    erbo_main_query_mode()
