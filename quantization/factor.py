import pandas as pd
import numpy as np
alpha = 0.5
factor_names = ['volume_price_turn_body']

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
    required_columns = ['open', 'close', 'volume','pctChg']
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
            df.iloc[i]['close'] > df.iloc[i]['open'] * 1.005  # 收盘价大于开盘价
            # df.iloc[i]['pctChg'] < 5  # 涨跌幅小于5
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
    IC均值: 0.0066, IC标准差: 0.0411, IR: 0.1600
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
    df['volume_support_times'] = df['volume_support'].rolling(window=20).sum()
    df = df.drop(columns=['continuous_shrink_small_positive', 'volume_support'])
    return df

# 成交量与价格背离
def mark_volume_price_divergence(df: pd.DataFrame):
    """
    判断成交量与价格的背离情况：
    - 如果当天成交量大于前一天，且收盘价小于前一天，则标记 'volume_price_divergence' 列为 1，否则为 0。
    - 统计 'volume_price_divergence' 列中值为 1 的次数，并记录在 'volume_price_divergence_times' 列中。
    :param df: 包含股票数据的 DataFrame，必须包含 '成交量' 和 '收盘价' 列。
    :return: 增加 'volume_price_divergence' 和 'volume_price_divergence_times' 列的 DataFrame。
    IC均值: -0.0357, IC标准差: 0.1116, IR: -0.3198 no log1p
    IC均值: -0.0369, IC标准差: 0.1148, IR: -0.3209 with log1p
    IC均值: -0.0464, IC标准差: 0.1222, IR: -0.3801 mean 10
     IC均值: -0.0474, IC标准差: 0.1394, IR: -0.3403 ma mean30 mean 10
    """
    # 初始化标记列
    df['volume_price_divergence'] = 0

    close_ma = df['close'].rolling(window=20).mean()
    volume_ma = df['volume'].rolling(window=20).mean()

    norm_volume = (df['volume'] - volume_ma)/volume_ma
    norm_close = (df['close'] - close_ma)/close_ma

    df['volume_price_divergence'] = abs(norm_close)/abs(norm_volume)
    df['volume_price_divergence'] = np.log1p(df['volume_price_divergence'])
    df['volume_price_divergence'] = np.where(norm_close > 0, df['volume_price_divergence'], -df['volume_price_divergence'])
    df['volume_price_divergence'] = pd.Series(df['volume_price_divergence']).rolling(window=10).mean()
    return df

def calculate_amihud_illiquidity(df, window=21):
    """
    计算Amihud非流动性因子
    
    参数:
    price_series: 股价序列 (pd.Series)
    volume_series: 成交量序列 (pd.Series)，与price_series索引相同
    window: 滚动窗口大小，默认为21个交易日（约1个月）
    
    返回:
    Amihud因子值序列
    IC均值: 0.0592, IC标准差: 0.1302, IR: 0.4547
    """
    required_columns = ['pctChg', 'amount']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必须包含 '{col}' 列！")
    # 计算日收益率
    daily_returns = df['pctChg']
    
    # 计算成交额（假设price_series是收盘价，volume_series是成交量）
    daily_turnover = df['amount']
    
    # 计算日度非流动性：|收益率| / 成交额
    # 添加一个极小值防止除零错误
    daily_illiquidity = np.abs(daily_returns) / (daily_turnover + 1e-10)
    
    # 计算滚动窗口平均值，并取对数平滑
    amihud_factor = daily_illiquidity.rolling(window=window).mean()
    amihud_factor_log = np.log(1 + amihud_factor)
    df['amihud_illiquidity'] = amihud_factor_log
    return df

# 量价换手分布
def calculate_volume_price_turnover_percentile(df: pd.DataFrame):
    """
    计算量价换手分布
    :param df: 包含股票数据的 DataFrame，必须包含 '成交量' 和 '收盘价' 列。
    :return: 增加 'volume_price_turnover' 列的 DataFrame。
     IC均值: 0.0543, IC标准差: 0.1107, IR: 0.4906
    """
    # 计算换手率在过去60天所处的分位数
    window = 60
    turn_percentile = 1/df['turn'].rolling(window=window).rank(pct=True)
    turn_percentile = np.log1p(turn_percentile)
    volume_percentile = 1/df['volume'].rolling(window=window).rank(pct=True)
    volume_percentile = np.log1p(volume_percentile)
    close_percentile = 1/df['close'].rolling(window=window).rank(pct=True)
    close_percentile = np.log1p(close_percentile)
    # k线实体分布
    body_percentile = 1/abs(df['close'] - df['open']).rolling(window=window).rank(pct=True)
    body_percentile = np.log1p(body_percentile)
    df['volume_price_turn_body'] = turn_percentile * body_percentile * close_percentile * volume_percentile/4
    df['volume_price_turn_body'] = df['volume_price_turn_body'].rolling(window=10).mean()
    return df

factor_dict = {
    'volume_support_times': mark_volume_support,
    'volume_price_divergence': mark_volume_price_divergence,
    'amihud_illiquidity': calculate_amihud_illiquidity,
    'volume_price_turn_body': calculate_volume_price_turnover_percentile,


}
mask_dict = {
    'volume_support_times': True,
    'volume_price_divergence': False,
    'amihud_illiquidity': False,
    'volume_price_turn_body': False,
}