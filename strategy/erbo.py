import pandas as pd
import warnings
import os
import sys
import numpy as np
from scipy import stats
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from init import get_specific_stocks_latest_data
# from init.data_prepare import read_main_board, read_growth_board, get_stock_zh_a_hist, get_start_to_end_date, get_stock_zh_a_hist_batch

alpha = 0.5  # 黄金分割率

def mark_zhu_erbo_condition1(df: pd.DataFrame, window=20):
    """
    朱二波条件1：
    标记最近 window 天内有涨跌幅大于 9.85%，并且当天满足以下条件：
    - 涨跌幅在 -1% 到 3% 之间。
    - 振幅小于 5%。
    - 换手率大于 2%。
    在 DataFrame 中增加一列 '朱二波条件1'，标记当天是否符合条件。
    :param df: 包含股票数据的 DataFrame，必须包含 '涨跌幅'、'振幅' 和 '换手率' 列
    :param window: 滑动窗口天数，默认10天
    :return: 增加标记列的 DataFrame
    """
    required_columns = ['涨跌幅', '振幅', '换手率']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    df['朱二波条件1'] = False

    for i in range(window - 1, len(df)):
        window_df = df.iloc[i - window + 1:i + 1]
        if (
            window_df['涨跌幅'].max() > 9.85 and
            -1 <= window_df.iloc[-1]['涨跌幅'] <= 3 and
            window_df.iloc[-1]['振幅'] < 5 and
            window_df.iloc[-1]['换手率'] > 2
        ):
            df.at[df.index[i], '朱二波条件1'] = True

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
    # 记录连续天数的最大值
    df['连续缩量小阳线最大值'] = df['连续缩量小阳线'].cummax()
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

def mark_yang_yin_difference(df: pd.DataFrame):
    """
    计算过去二十天涨跌幅在-5%到5%之间阳线比阴线多多少天：
    - 过去二十天中涨跌幅在-5%到5%之间的交易日
    - 阳线：收盘价大于开盘价
    - 阴线：收盘价小于开盘价
    - 计算阳线天数减去阴线天数的差值
    在 DataFrame 中增加一列 '阳线比阴线多天数'，记录差值。
    :param df: 包含股票数据的 DataFrame，必须包含 '开盘'、'收盘'、'涨跌幅' 列
    :return: 增加标记列的 DataFrame
    """
    required_columns = ['开盘', '收盘', '涨跌幅']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    # 初始化标记列
    df['阳线比阴线多天数'] = 0

    # 滑动窗口处理
    for i in range(len(df)):
        # 获取过去二十天的窗口（包括当天）
        window = df.iloc[max(0, i - 19):i + 1]
        
        # 筛选涨跌幅在-5%到5%之间的交易日
        filtered_window = window[(window['涨跌幅'] >= -5) & (window['涨跌幅'] <= 5)]
        
        if len(filtered_window) == 0:
            df.at[df.index[i], '阳线比阴线多天数'] = 0
            continue
        
        # 计算阳线和阴线天数
        yang_days = (filtered_window['收盘'] > filtered_window['开盘']).sum()  # 阳线：收盘 > 开盘
        yin_days = (filtered_window['收盘'] < filtered_window['开盘']).sum()   # 阴线：收盘 < 开盘
        
        # 计算差值
        difference = yang_days - yin_days
        df.at[df.index[i], '阳线比阴线多天数'] = difference

    return df

def mark_low_volatility_and_flat_trend(df: pd.DataFrame, window_days=5):
    """
    判断最近几天低波动且趋势平缓的条件：
    - 最近五天的最大涨幅小于5%
    - 最近几天的收盘价用一条线拟合后斜率小于1
    在 DataFrame 中增加一列 '低波动平缓趋势'，标记当天是否符合条件。
    同时增加 '最近5天最大涨幅' 和 '收盘价拟合斜率' 列用于分析。
    :param df: 包含股票数据的 DataFrame，必须包含 '涨跌幅' 和 '收盘' 列
    :param window_days: 分析的天数窗口，默认5天
    :return: 增加标记列的 DataFrame
    """
    required_columns = ['涨跌幅', '收盘']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    # 初始化标记列
    df['低波动平缓趋势'] = 0
    df['最近5天最大涨幅'] = 0.0
    df['收盘价拟合斜率'] = 0.0

    # 滑动窗口处理
    for i in range(len(df)):
        # 获取最近window_days天的窗口（包括当天）
        start_idx = max(0, i - window_days + 1)
        window = df.iloc[start_idx:i + 1]
        
        if len(window) < 2:  # 至少需要2个点才能拟合直线
            continue
            
        # 1. 计算最近五天的最大涨幅
        max_gain = window['涨跌幅'].max()
        df.at[df.index[i], '最近5天最大涨幅'] = max_gain
        
        # 2. 对收盘价进行线性拟合
        # 创建x轴数据（天数索引）
        # x = np.arange(len(window))
        # y = window['收盘'].values
        
        # 使用最小二乘法进行线性拟合
        # try:
        #     slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        #     df.at[df.index[i], '收盘价拟合斜率'] = slope
            
            # 3. 判断是否同时满足两个条件
        if max_gain < 5.0:
            df.at[df.index[i], '低波动平缓趋势'] = 1
                
        # except Exception as e:
        #     # 如果拟合失败，记录为0
        #     df.at[df.index[i], '收盘价拟合斜率'] = 0

    return df

def mark_market_protection(df: pd.DataFrame):
    """
    判断护盘情况：
    - 过去二十天是否有涨幅大于3%的阳线且小于5
    - 如果有，判断第二天是否是涨幅大于0%的阳线并且缩量
    - 如果是，则标记为护盘
    - 统计护盘次数
    在 DataFrame 中增加两列：
    - '护盘'：当天是否为护盘（0/1）
    - '护盘次数'：累计护盘次数
    :param df: 包含股票数据的 DataFrame，必须包含 '涨跌幅'、'开盘'、'收盘'、'成交量' 列
    :return: 增加标记列的 DataFrame
    """
    required_columns = ['涨跌幅', '开盘', '收盘', '成交量']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    # 初始化标记列
    df['护盘'] = 0
    df['护盘次数'] = 0

    # 滑动窗口处理
    for i in range(1, len(df)):  # 从第1行开始，确保有前一天的数据
        # 获取过去二十天的窗口（不包括当天，因为要判断前一天的情况）
        start_idx = max(0, i - 20)
        window = df.iloc[start_idx:i]  # 过去二十天（不包括当天）
        
        if len(window) == 0:
            continue
            
        # 1. 检查过去二十天是否有涨幅大于3%且小于5%的阳线
        # 阳线条件：收盘价大于开盘价
        yang_lines = window[(window['收盘'] > window['开盘']) & (window['涨跌幅'] > 3)
                            & (window['涨跌幅'] < 5)]
        
        if len(yang_lines) == 0:
            continue
            
        # 2. 找到最近一次涨幅大于3%的阳线
        last_big_yang_idx = yang_lines.index[-1]  # 最后一次大阳线的索引
        
        # 3. 检查是否是前一天，并且当天（第二天）符合护盘条件
        if last_big_yang_idx == df.index[i-1]:  # 前一天是大阳线
            current_day = df.iloc[i]
            previous_day = df.iloc[i-1]
            
            # 护盘条件：
            # - 当天涨幅大于0%
            # - 当天是阳线（收盘价大于开盘价）
            # - 当天缩量（成交量小于前一天）
            if (current_day['涨跌幅'] > 0 and 
                current_day['涨跌幅'] < 5 and
                current_day['收盘'] > current_day['开盘'] and 
                current_day['成交量'] < previous_day['成交量']):
                
                df.at[df.index[i], '护盘'] = 1

    # 统计护盘次数（累计）
    df['护盘次数'] = df['护盘'].cumsum()

    return df

def mark_volume_shrinkage_vs_expansion(df: pd.DataFrame):
    """
    计算过去二十天缩量天数比放量天数多多少天：
    - 缩量：当天成交量比前一天小
    - 放量：当天成交量比前一天大
    - 计算缩量天数减去放量天数的差值
    在 DataFrame 中增加一列 '缩量比放量多天数'，记录差值。
    :param df: 包含股票数据的 DataFrame，必须包含 '成交量' 列
    :return: 增加标记列的 DataFrame
    """
    required_columns = ['成交量']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    # 初始化标记列
    df['缩量比放量多天数'] = 0

    # 滑动窗口处理
    for i in range(1, len(df)):  # 从第1行开始，因为需要和前一天比较
        # 获取过去二十天的窗口（包括当天）
        start_idx = max(1, i - 19)  # 确保至少从第1行开始
        window = df.iloc[start_idx:i + 1]
        
        if len(window) < 2:  # 至少需要2天数据才能比较
            continue
        
        # 计算缩量和放量天数
        shrinkage_days = 0  # 缩量天数
        expansion_days = 0  # 放量天数
        
        for j in range(1, len(window)):  # 从窗口的第二天开始比较
            current_volume = window.iloc[j]['成交量']
            previous_volume = window.iloc[j-1]['成交量']
            
            if current_volume < previous_volume:
                shrinkage_days += 1  # 缩量
            elif current_volume > previous_volume:
                expansion_days += 1  # 放量
            # 成交量相等的情况不计入缩量或放量
        
        # 计算差值：缩量天数减去放量天数
        difference = shrinkage_days - expansion_days
        df.at[df.index[i], '缩量比放量多天数'] = difference

    return df

def mark_drop_rise_vs_drop_drop(df: pd.DataFrame):
    """
    统计最近二十天第一天跌第二天涨比第一天跌第二天继续跌的天数多多少：
    - 跌涨模式：第一天涨跌幅为负，第二天涨跌幅为正
    - 跌跌模式：第一天涨跌幅为负，第二天涨跌幅也为负
    - 计算跌涨天数减去跌跌天数的差值
    在 DataFrame 中增加一列 '跌涨比跌跌多天数'，记录差值。
    :param df: 包含股票数据的 DataFrame，必须包含 '涨跌幅' 列
    :return: 增加标记列的 DataFrame
    """
    required_columns = ['涨跌幅']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    # 初始化标记列
    df['跌涨比跌跌多天数'] = 0

    # 滑动窗口处理
    for i in range(1, len(df)):  # 从第1行开始，因为需要比较前一天
        # 获取过去二十天的窗口（包括当天）
        start_idx = max(1, i - 19)  # 确保至少从第1行开始
        window = df.iloc[start_idx:i + 1]
        
        if len(window) < 2:  # 至少需要2天数据才能比较
            continue
        
        # 计算跌涨和跌跌模式的天数
        drop_rise_days = 0  # 跌涨天数
        drop_drop_days = 0  # 跌跌天数
        
        for j in range(1, len(window)):  # 从窗口的第二天开始比较
            first_day_change = window.iloc[j-1]['涨跌幅']
            second_day_change = window.iloc[j]['涨跌幅']
            
            if first_day_change < 0:  # 第一天跌
                if second_day_change > 0:  # 第二天涨
                    drop_rise_days += 1
                elif second_day_change < 0:  # 第二天继续跌
                    drop_drop_days += 1
                # 第二天平盘（涨跌幅为0）不计入统计
        
        # 计算差值：跌涨天数减去跌跌天数
        difference = drop_rise_days - drop_drop_days
        df.at[df.index[i], '跌涨比跌跌多天数'] = difference

    return df

def mark_wash_trading(df: pd.DataFrame):
    """
    判断洗盘情况：
    - 第一天是阴线且跌幅大于2%
    - 后面两天内价格回到第一天的开盘价
    - 如果满足条件则标记为洗盘
    - 统计洗盘次数
    在 DataFrame 中增加两列：
    - '洗盘'：当天是否为洗盘完成（0/1）
    - '洗盘次数'：累计洗盘次数
    :param df: 包含股票数据的 DataFrame，必须包含 '涨跌幅'、'开盘'、'收盘'、'最高' 列
    :return: 增加标记列的 DataFrame
    """
    required_columns = ['涨跌幅', '开盘', '收盘', '最高']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    # 初始化标记列
    df['洗盘'] = 0
    df['洗盘次数'] = 0

    # 遍历数据，至少需要3天数据（第一天+后面两天）
    for i in range(len(df) - 2):
        first_day = df.iloc[i]
        second_day = df.iloc[i + 1]
        third_day = df.iloc[i + 2]
        
        # 1. 判断第一天是否为阴线且跌幅大于2%
        is_first_day_qualified = (
            first_day['收盘'] < first_day['开盘'] and  # 阴线：收盘价小于开盘价
            first_day['涨跌幅'] < -2  # 跌幅大于2%（涨跌幅为负值）
        )
        
        if not is_first_day_qualified:
            continue
            
        # 2. 检查后面两天内价格是否回到第一天的开盘价
        first_day_open = first_day['开盘']
        
        # 检查第二天或第三天的最高价是否达到或超过第一天的开盘价
        second_day_recovered = second_day['最高'] >= first_day_open
        third_day_recovered = third_day['最高'] >= first_day_open
        
        # 如果后面两天内任意一天回到第一天开盘价，则标记为洗盘
        if second_day_recovered:
            # 在第二天标记洗盘完成
            df.at[df.index[i + 1], '洗盘'] = 1
        elif third_day_recovered:
            # 在第三天标记洗盘完成
            df.at[df.index[i + 2], '洗盘'] = 1

    # 统计洗盘次数（累计）
    df['洗盘次数'] = df['洗盘'].cumsum()

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
    results=[]
    df_dict = get_specific_stocks_latest_data()

    for code, name_df in df_dict.items():
        name = name_df['name']
        df = name_df['data']
        # 替换df的英文列名为中文名（如果需要）
        column_map = {
            'open': '开盘',
            'close': '收盘',
            'high': '最高',
            'low': '最低',
            'volume': '成交量',
            'pctChg': '涨跌幅',
            'turn': '换手率'
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        # 利用昨天的收盘价计算振幅（如果没有“振幅”列则补充）
        if '振幅' not in df.columns:
            if '最高' in df.columns and '最低' in df.columns and '收盘' in df.columns:
                df['振幅'] = ((df['最高'] - df['最低']) / df['收盘'].shift(1).replace(0, np.nan)) * 100
                df['振幅'] = df['振幅'].fillna(0)  # 处理NaN值
            else:
                raise ValueError("DataFrame 缺少计算振幅所需的列（最高、最低、收盘）")
        # 3. 应用所有mark方法
        df = mark_zhu_erbo_condition1(df)
        df = mark_consecutive_small_positive(df)
        df = mark_volume_support(df)
        df = mark_yang_yin_difference(df)
        df = mark_low_volatility_and_flat_trend(df)
        df = mark_market_protection(df)
        df = mark_volume_shrinkage_vs_expansion(df)
        df = mark_drop_rise_vs_drop_drop(df)




        # 4. 统计最新一天命中条件数量和具体命中条件
        last = df.iloc[-1]
        small_positive_days = int(last.get('连续缩量小阳线', 0))
        volume_support_count = int(last.get('缩量承接次数', 0))
        is_volume_support = bool(last.get('缩量承接', 0))
        yang_yin_difference = int(last.get('阳线比阴线多天数', 0))
        zhuerbo = int(last.get('朱二波条件1', 0))
        consecutive_small_positive = int(last.get('连续缩量小阳线最大值', 0))
        low_volatility_and_flat_trend = int(last.get('低波动平缓趋势', 0))
        market_protection = int(last.get('护盘次数', 0))
        volume_shrinkage_vs_expansion = int(last.get('缩量比放量多天数', 0))
        drop_rise_vs_drop_drop = int(last.get('跌涨比跌跌多天数', 0))
        volume_support_and_market_protection = volume_support_count + market_protection
        wash_trading = int(last.get('洗盘', 0))

        results.append({
        "代码": code,
        "名称": name,
        "连续缩量小阳线天数": small_positive_days,
        "连续缩量小阳线最大值": consecutive_small_positive,
        "今天是否为缩量承接": 1 if is_volume_support else 0,
        "缩量承接次数": volume_support_count,
        '阳线比阴线多天数': yang_yin_difference,
        '二波': zhuerbo,
        '低波动平缓趋势': low_volatility_and_flat_trend,
        '护盘次数': market_protection,
        '缩量比放量多天数': volume_shrinkage_vs_expansion,
        '跌涨比跌跌多天数': drop_rise_vs_drop_drop,
        '缩量承接加护盘次数': volume_support_and_market_protection,
        '洗盘': wash_trading,
        '洗盘承接护盘次数': wash_trading + market_protection + volume_support_count


        })

    #  将结果存入 Excel 文件
    result_df = pd.DataFrame(results)
    result_df.to_excel("result.xlsx", index=False)
    print("结果已保存到 result.xlsx 文件中。")

if __name__ == "__main__":
    erbo_main_query_mode()
