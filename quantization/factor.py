import pandas as pd
alpha = 0.5
factor_names = ['volume_support_times']
def _mark_consecutive_small_positive(df: pd.DataFrame):
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
    required_columns = ['open', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")

    # 初始化标记列
    df['continuous_shrink_small_positive'] = 0

    # 滑动窗口处理
    consecutive_days = 0  # 记录连续天数
    for i in range(len(df)):
        # 计算最近 20 天成交量的均值和标准差
        volume_window = df.iloc[max(0, i - 19):i + 1]  # 最近 20 天的窗口
        volume_mean = volume_window['volume'].mean()
        volume_std = volume_window['volume'].std()

        # 计算最近 20 天开盘减去收盘绝对值的均值和标准差
        open_close_diff = (volume_window['open'] - volume_window['close']).abs()  # 开盘减去收盘的绝对值
        diff_mean = open_close_diff.mean()
        diff_std = open_close_diff.std()

        # 判断当天是否符合条件
        current_diff = abs(df.iloc[i]['open'] - df.iloc[i]['close'])
        if (
            df.iloc[i]['volume'] < (volume_mean - alpha * volume_std) and  # 成交量条件
            current_diff < (diff_mean - alpha * diff_std) and  # 开盘减去收盘绝对值条件
            df.iloc[i]['close'] > df.iloc[i]['open']  # 收盘价大于开盘价
        ):
            consecutive_days += 1
            df.at[df.index[i], 'continuous_shrink_small_positive'] = consecutive_days
        else:
            consecutive_days = 0  # 不符合条件时重置连续天数
    return df

def mark_volume_support(df: pd.DataFrame):
    """
    判断缩量承接：
    - 如果当天连续缩量小阳线天数大于等于 2，且当天成交量比前一天小，则标记 '缩量承接' 列为 1，否则为 0。
    - 统计 '缩量承接' 列中值为 1 的次数，并记录在 '缩量承接次数' 列中。
    :param df: 包含股票数据的 DataFrame，必须包含 '连续缩量小阳线' 和 '成交量' 列。
    :return: 增加 '缩量承接' 和 '缩量承接次数' 列的 DataFrame。
    """

    # 初始化标记列
    df = _mark_consecutive_small_positive(df)
    df['volume_support'] = 0

    # 遍历 DataFrame，检查条件
    for i in range(1, len(df)):  # 从第 1 行开始，确保有前一天的数据
        if (
            df.iloc[i]['continuous_shrink_small_positive'] >= 2 and  # 当天连续缩量小阳线天数大于等于 2
            df.iloc[i]['volume'] < df.iloc[i - 1]['volume']  # 当天成交量比前一天小
        ):
            df.at[df.index[i], 'volume_support'] = 1

    # 统计最近 20 天 'volume_support' 列中值为 1 的次数
    df['volume_support_times'] = df['volume_support'].rolling(window=20, min_periods=1).sum()

    return df