import pandas as pd
import warnings
from init import *
from init.data_prepare import read_main_board, read_growth_board, get_stock_zh_a_hist, get_start_to_end_date, get_stock_zh_a_hist_batch

alpha = 0.5  # 黄金分割率
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
    print(f"筛选后的股票数量: {len(filtered_data)}")
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
    - 最近 20 天成交量的均值和标准差。
    - 成交量小于均值减标准差。
    - 最近 20 天开盘减去收盘绝对值的均值和标准差。
    - 开盘减去收盘的绝对值小于均值减去 1 个标准差且收盘价大于开盘价。
    - 统计符合条件的连续天数，并记录在一列 '连续缩量小阳线' 中。
    :param df: 包含股票数据的 DataFrame，必须包含 '开盘'、'收盘'、'最高'、'最低' 和 '成交量' 列。
    :return: 增加标记列的 DataFrame。
    """
    required_columns = ['开盘', '收盘', '成交量']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    # 初始化标记列
    df['连续缩量小阳线'] = 0

    # 滑动窗口处理
    consecutive_days = 0  # 记录连续天数
    for i in range(len(df)):
        # 计算最近 20 天成交量的均值和标准差
        volume_window = df.iloc[max(0, i - 19):i + 1]  # 最近 20 天的窗口
        volume_mean = volume_window['成交量'].mean()
        volume_std = volume_window['成交量'].std()

        # 计算最近 20 天开盘减去收盘绝对值的均值和标准差
        open_close_diff = (volume_window['开盘'] - volume_window['收盘']).abs()  # 开盘减去收盘的绝对值
        diff_mean = open_close_diff.mean()
        diff_std = open_close_diff.std()

        # 判断当天是否符合条件
        current_diff = abs(df.iloc[i]['开盘'] - df.iloc[i]['收盘'])
        if (
            df.iloc[i]['成交量'] < (volume_mean - alpha * volume_std) and  # 成交量条件
            current_diff < (diff_mean - alpha * diff_std) and  # 开盘减去收盘绝对值条件
            df.iloc[i]['收盘'] > df.iloc[i]['开盘']  # 收盘价大于开盘价
        ):
            consecutive_days += 1
            df.at[df.index[i], '连续缩量小阳线'] = consecutive_days
        else:
            consecutive_days = 0  # 不符合条件时重置连续天数

    return df

def mark_consecutive_support(df: pd.DataFrame):
    """
    检查过去连续缩量小阳线的值：
    - 如果当天的 '连续缩量小阳线' 值大于等于 2 且后一天的值小于今天的值，则标记 '承接' 列为 1，否则为 0。
    - 统计 '承接' 列中值为 1 的次数，并记录在 '总共承接次数' 列中。
    :param df: 包含股票数据的 DataFrame，必须包含 '连续缩量小阳线' 列。
    :return: 增加 '承接' 和 '总共承接次数' 列的 DataFrame。
    """
    if '连续缩量小阳线' not in df.columns:
        raise ValueError("DataFrame 必须包含 '连续缩量小阳线' 列！")

    # 初始化标记列
    df['承接'] = 0

    # 遍历 DataFrame，检查条件
    for i in range(len(df) - 1):  # 确保不会超出索引范围
        if df.iloc[i]['连续缩量小阳线'] >= 2 and df.iloc[i + 1]['连续缩量小阳线'] < df.iloc[i]['连续缩量小阳线']:
            df.at[df.index[i], '承接'] = 1

    # 统计 '承接' 列中值为 1 的次数
    df['总共承接次数'] = df['承接'].cumsum()

    return df

def mark_volume_support(df: pd.DataFrame):
    """
    判断缩量承接：
    - 如果当天连续缩量小阳线天数大于等于 2，且当天成交量比前一天小，则标记 '缩量承接' 列为 1，否则为 0。
    - 统计 '缩量承接' 列中值为 1 的次数，并记录在 '缩量承接次数' 列中。
    :param df: 包含股票数据的 DataFrame，必须包含 '连续缩量小阳线' 和 '成交量' 列。
    :return: 增加 '缩量承接' 和 '缩量承接次数' 列的 DataFrame。
    """
    required_columns = ['连续缩量小阳线', '成交量']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    # 初始化标记列
    df['缩量承接'] = 0

    # 遍历 DataFrame，检查条件
    for i in range(1, len(df)):  # 从第 1 行开始，确保有前一天的数据
        if (
            df.iloc[i]['连续缩量小阳线'] >= 2 and  # 当天连续缩量小阳线天数大于等于 2
            df.iloc[i]['成交量'] < df.iloc[i - 1]['成交量']  # 当天成交量比前一天小
        ):
            df.at[df.index[i], '缩量承接'] = 1

    # 统计 '缩量承接' 列中值为 1 的次数
    df['缩量承接次数'] = df['缩量承接'].cumsum()

    return df

def mark_consecutive_small_negative(df: pd.DataFrame):
    """
    寻找连续的缩量小阴线：
    - 最近 20 天成交量的均值和标准差。
    - 成交量小于均值减标准差。
    - 最近 20 天开盘减去收盘绝对值的均值和标准差。
    - 开盘减去收盘的绝对值小于均值减去 1 个标准差且收盘价小于开盘价。
    - 统计符合条件的连续天数，并记录在一列 '连续缩量小阴线' 中。
    :param df: 包含股票数据的 DataFrame，必须包含 '开盘'、'收盘' 和 '成交量' 列。
    :return: 增加标记列的 DataFrame。
    """
    required_columns = ['开盘', '收盘', '成交量']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    # 初始化标记列
    df['连续缩量小阴线'] = 0

    # 滑动窗口处理
    consecutive_days = 0  # 记录连续天数
    for i in range(len(df)):
        # 计算最近 20 天成交量的均值和标准差
        volume_window = df.iloc[max(0, i - 19):i + 1]  # 最近 20 天的窗口
        volume_mean = volume_window['成交量'].mean()
        volume_std = volume_window['成交量'].std()

        # 计算最近 20 天开盘减去收盘绝对值的均值和标准差
        open_close_diff = (volume_window['开盘'] - volume_window['收盘']).abs()  # 开盘减去收盘的绝对值
        diff_mean = open_close_diff.mean()
        diff_std = open_close_diff.std()

        # 判断当天是否符合条件
        current_diff = abs(df.iloc[i]['开盘'] - df.iloc[i]['收盘'])
        if (
            df.iloc[i]['成交量'] < (volume_mean - alpha * volume_std) and  # 成交量条件
            current_diff < (diff_mean - alpha * diff_std) and  # 开盘减去收盘绝对值条件
            df.iloc[i]['收盘'] < df.iloc[i]['开盘']  # 收盘价小于开盘价（小阴线）
        ):
            consecutive_days += 1
            df.at[df.index[i], '连续缩量小阴线'] = consecutive_days
        else:
            consecutive_days = 0  # 不符合条件时重置连续天数

    return df

def mark_two_positive_one_negative(df: pd.DataFrame):
    """
    判断两阳夹一阴的情况：
    - 如果当天连续缩量小阳线为 1 天，前一天连续缩量小阴线为 1 天，再前一天连续缩量小阳线为 1 天，
      则标记 '两阳夹一阴' 列为 1，否则为 0。
    - 统计 '两阳夹一阴' 列中值为 1 的次数，并记录在 '总共两阳夹一阴次数' 列中。
    :param df: 包含股票数据的 DataFrame，必须包含 '连续缩量小阳线' 和 '连续缩量小阴线' 列。
    :return: 增加 '两阳夹一阴' 和 '总共两阳夹一阴次数' 列的 DataFrame。
    """
    required_columns = ['连续缩量小阳线', '连续缩量小阴线']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    # 初始化标记列
    df['两阳夹一阴'] = 0

    # 遍历 DataFrame，检查条件
    for i in range(2, len(df)):  # 从第 2 行开始，确保有足够的前一天和再前一天数据
        if (
            df.iloc[i]['连续缩量小阳线'] == 1 and  # 当天为小阳线 1 天
            df.iloc[i - 1]['连续缩量小阴线'] == 1 and  # 前一天为小阴线 1 天
            df.iloc[i - 2]['连续缩量小阳线'] == 1  # 再前一天为小阳线 1 天
        ):
            df.at[df.index[i], '两阳夹一阴'] = 1

    # 统计 '两阳夹一阴' 列中值为 1 的次数
    df['总共两阳夹一阴次数'] = df['两阳夹一阴'].cumsum()

    return df

def mark_lower_shadow_analysis(df: pd.DataFrame):
    """
    统计过去 20 天的下影线平均长度，并计算最近 10 天下影线大于均值的天数。
    - 下影线长度 = 开盘价和收盘价的较小值减去最低价。
    - 过去 20 天计算下影线的平均长度。
    - 统计最近 10 天中下影线长度大于过去 20 天均值的天数。
    在 DataFrame 中增加两列：
    - '下影线均值'：过去 20 天下影线的平均长度。
    - '最近10天下影线大于均值天数'：最近 10 天中下影线大于均值的天数。
    :param df: 包含股票数据的 DataFrame，必须包含 '开盘'、'收盘' 和 '最低' 列。
    :return: 增加标记列的 DataFrame。
    """
    required_columns = ['开盘', '收盘', '最低']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    # 初始化标记列
    df['下影线均值'] = 0.0
    df['最近10天下影线大于均值天数'] = 0

    # 计算下影线长度
    df['下影线长度'] = df[['开盘', '收盘']].min(axis=1) - df['最低']

    # 滑动窗口处理
    for i in range(len(df)):
        # 计算过去 20 天的下影线均值
        lower_shadow_window = df.iloc[max(0, i - 19):i + 1]  # 最近 20 天的窗口
        lower_shadow_mean = lower_shadow_window['下影线长度'].mean()
        df.at[df.index[i], '下影线均值'] = lower_shadow_mean

        # 计算最近 10 天中下影线大于均值的天数
        if i >= 9:  # 确保有足够的 10 天数据
            recent_10_days = df.iloc[i - 9:i + 1]  # 最近 10 天的窗口
            count_greater_than_mean = (recent_10_days['下影线长度'] > lower_shadow_mean).sum()
            df.at[df.index[i], '最近10天下影线大于均值天数'] = count_greater_than_mean

    # 删除临时列
    df.drop(columns=['下影线长度'], inplace=True)

    return df

def erbo_main_query_mode():
    """
    二波主函数-查询模式：
    1. 过滤股票名单
    2. 查询最近30天历史行情（批量获取）
    3. 应用所有mark方法
    4. 统计最新一天命中条件数量，按照缩量小阳线天数从多到少排序打印
    5. 将结果存入 result.xlsx 文件
    """
    # 1. 过滤股票名单
    filtered_stocks = filter_zhu_erbo_condition0()
    if not filtered_stocks:
        print("没有符合条件的股票。")
        return

    # 构造参数列表
    params_list = []
    start_date, end_date = get_start_to_end_date(32)
    for code, name, _ in filtered_stocks:
        params_list.append({
            "symbol": code,
            "period": "daily",
            "start_date": start_date,
            "end_date": end_date,
            "adjust": "qfq"  # 可根据需求调整复权方式
        })

    # 2. 批量查询最近30天历史行情
    result_data = get_stock_zh_a_hist_batch(params_list)

    results = []
    for (params, df), (_, name, _) in zip(result_data, filtered_stocks):
        code = params["symbol"]

        if df.empty:
            print(f"{code} {name} 获取数据失败，跳过。")
            continue

        # 3. 应用所有mark方法
        df = mark_long_erbo_condition1(df)
        df = mark_short_erbo_condition1(df)
        df = mark_zhu_erbo_condition1(df)
        df = mark_consecutive_small_positive(df)
        df = mark_consecutive_small_negative(df)
        df = mark_consecutive_support(df)
        df = mark_volume_support(df)
        df = mark_two_positive_one_negative(df)
        df = mark_lower_shadow_analysis(df)

        # # 将 DataFrame 存入 Excel 文件
        # df.to_excel("table.xlsx", index=False)
        # print(f"{code} {name} 的数据已保存到 table.xlsx 文件中。")

        # 4. 统计最新一天命中条件数量和具体命中条件
        last = df.iloc[-1]
        small_positive_days = int(last.get('连续缩量小阳线', 0))
        total_support_count = int(last.get('总共承接次数', 0))
        volume_support_count = int(last.get('缩量承接次数', 0))
        is_volume_support = bool(last.get('缩量承接', 0))
        total_two_positive_one_negative_count = int(last.get('总共两阳夹一阴次数', 0))
        is_two_positive_one_negative = bool(last.get('两阳夹一阴', 0))
        lower_shadow_days = int(last.get('最近10天下影线大于均值天数', 0))  # 下影线条件
        conditions = []
        if last.get('长二波条件1', False):
            conditions.append("长二波条件1")
        if last.get('短二波条件1', False):
            conditions.append("短二波条件1")
        if last.get('朱二波条件1', False):
            conditions.append("朱二波条件1")
        if small_positive_days > 0:
            conditions.append(f"连续缩量小阳线{small_positive_days}天")
        if lower_shadow_days > 0:
            conditions.append(f"最近10天下影线大于均值{lower_shadow_days}天")

        if conditions:
            results.append({
            "代码": code,
            "名称": name,
            "连续缩量小阳线天数": small_positive_days,
            "今天是否两阳夹一阴": "是" if is_two_positive_one_negative else "否",
            "今天是否为缩量承接": "是" if is_volume_support else "否",
            "总共承接次数": total_support_count,
            "缩量承接次数": volume_support_count,
            "总共两阳夹一阴次数": total_two_positive_one_negative_count,
            "最近10天下影线大于均值天数": lower_shadow_days,
            "长二波条件1": last.get('长二波条件1', False),
            "短二波条件1": last.get('短二波条件1', False),
            "朱二波条件1": last.get('朱二波条件1', False),
            "命中条件": ", ".join(conditions),
            "承接+两阳一阴": total_support_count + total_two_positive_one_negative_count
            })

    # 5. 按照缩量小阳线天数从多到少排序
    results.sort(key=lambda x: (-x['连续缩量小阳线天数'], x['代码']))

        # 6. 打印结果并存入 Excel 文件
        # if results:
        #     print("命中条件数量排序结果：")
        # for r in results:
        #     print(
        #     f"{r['名称']}({r['代码']}): "
        #     f"连续缩量小阳线{r['连续缩量小阳线天数']}天, "
        #     f"今天是否两阳夹一阴: {r['今天是否两阳夹一阴']}, "
        #     f"总共承接次数: {r['总共承接次数']}, "
        #     f"总共两阳夹一阴次数: {r['总共两阳夹一阴次数']}, "
        #     f"最近10天下影线大于均值天数: {r['最近10天下影线大于均值天数']}, "
        #     f"长二波条件1: {r['长二波条件1']}, "
        #     f"短二波条件1: {r['短二波条件1']}, "
        #     f"朱二波条件1: {r['朱二波条件1']}, "
        #     f"命中条件: {r['命中条件']}"
        #     )

        # 将结果存入 Excel 文件
    result_df = pd.DataFrame(results)
    result_df.to_excel("result.xlsx", index=False)
    print("结果已保存到 result.xlsx 文件中。")
        # else:
        # print("没有符合条件的股票。")

if __name__ == "__main__":
    erbo_main_query_mode()
