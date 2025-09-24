import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_prepare.prepare import get_stock_merge_industry_table

import logging
import datetime
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"factor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def groupby_code(func):
    """
    装饰器：自动在含有'code'列的大表上按股票分组批量计算。
    """
    def wrapper(df, *args, **kwargs):
        if 'code' in df.columns:
            return df.groupby('code', group_keys=False, dropna=True).apply(lambda g: func(g, *args, **kwargs))
        else:
            return func(df, *args, **kwargs)
    return wrapper

alpha = 0.5

@groupby_code
def mark_volume_price_divergence(df: pd.DataFrame):
    """
    在大表（含多股票）上分组计算成交量与价格背离因子。
    """
    df = df.copy()
    close_ma = df['close'].rolling(window=20,min_periods=15).mean()
    volume_ma = df['volume'].rolling(window=20,min_periods=15).mean()
    norm_volume = (df['volume'] - volume_ma)/volume_ma
    norm_close = (df['close'] - close_ma)/close_ma
    df['volume_price_divergence'] = abs(norm_close)/abs(norm_volume)
    df['volume_price_divergence'] = np.log1p(df['volume_price_divergence'])
    df['volume_price_divergence'] = np.where(norm_close < 0, df['volume_price_divergence'], -df['volume_price_divergence'])
    df['volume_price_divergence'] = df['volume_price_divergence'].rolling(window=10, min_periods=8).mean()
    return df

@groupby_code
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
    市值中性化：IC均值: 0.0475, IC标准差: 0.1081, IR: 0.4400
    行业中性化: IC均值: 0.0395, IC标准差: 0.0738, IR: 0.5357
    """
    df = df.copy()
    daily_returns = df['pctChg']
    daily_turnover = df['amount']
    daily_illiquidity = np.abs(daily_returns) / (daily_turnover + 1e-10)
    amihud_factor = daily_illiquidity.rolling(window=window, min_periods=window-5).mean()
    amihud_factor_log = np.log(1 + amihud_factor)
    df['amihud_illiquidity'] = amihud_factor_log
    return df

@groupby_code
def calculate_volume_price_turnover_percentile(df: pd.DataFrame):
    """
    计算量价换手分布
    :param df: 包含股票数据的 DataFrame，必须包含 '成交量' 和 '收盘价' 列。
    :return: 增加 'volume_price_turnover' 列的 DataFrame。
     IC均值: 0.0543, IC标准差: 0.1107, IR: 0.4906
     去掉价格 IC均值: 0.0547, IC标准差: 0.0990, IR: 0.5522
     加入市值中性化 IC均值: 0.0569, IC标准差: 0.0949, IR: 0.5998
     行业中性化 IC均值: 0.0453, IC标准差: 0.0520, IR: 0.8713
     use 去除换手 IC均值: 0.0533, IC标准差: 0.0576, IR: 0.9260
     只有量 IC均值: 0.0510, IC标准差: 0.0580, IR: 0.8801
     只有k线 IC均值: 0.0495, IC标准差: 0.0495, IR: 1.0002
    """
    # 计算换手率在过去60天所处的分位数
    df = df.copy()
    window = 60
    turn_percentile = 1/df['turn'].rolling(window=window, min_periods=window-5).rank(pct=True)
    turn_percentile = np.log1p(turn_percentile)
    volume_percentile = 1/df['volume'].rolling(window=window, min_periods=window-5).rank(pct=True)
    volume_percentile = np.log1p(volume_percentile)
    body_percentile = 1/abs(df['close'] - df['open']).rolling(window=window, min_periods=window-5).rank(pct=True)
    body_percentile = np.log1p(body_percentile)
    df['volume_price_turn_body'] = body_percentile * volume_percentile
    df['volume_price_turn_body'] = df['volume_price_turn_body'].rolling(window=10, min_periods=8).mean()
    return df

@groupby_code
def calculate_volume_price_volatility(df: pd.DataFrame):
    """
    计算量价波动率
    :param df: 包含股票数据的 DataFrame，必须包含 '成交量' 和 '收盘价' 列。
    :return: 增加 'volume_price_volatility' 列的 DataFrame。
    IC均值: 0.0723, IC标准差: 0.0618, IR: 1.1712
    """
    df = df.copy()
    window = 60
    volume_percent = df['volume'].rolling(window=window,min_periods=window-5).rank(pct=True, ascending=False)
    volatility = df['close'].rolling(window=window,min_periods=window-5).std()/df['close'].rolling(window=window,min_periods=window-5).mean()
    volatility = np.log1p(1/volatility)
    df['volume_price_volatility'] = volume_percent * volatility
    df['volume_price_volatility'] = df['volume_price_volatility'].rolling(window=10, min_periods=8).mean()
    return df

def calculate_market_mean_return(df: pd.DataFrame):
    """
    计算全市场日均涨幅因子（所有股票每日涨幅均值，赋值到每只股票的对应日期行）
    :param df: 包含股票数据的 DataFrame，需包含 'pctChg' 和 'date' 列
    :return: 增加 'market_mean_return' 列的 DataFrame
    """
    df = df.copy()
    mean_return = df.groupby('date')['pctChg'].transform('mean')
    df['market_mean_return'] = mean_return
    return df

def calculate_market_mean_return_ma10_minus_ma5(df: pd.DataFrame):
    """
    计算 market_mean_return 的10日均线减去5日均线（全市场截面，所有股票每天该值一致）
    :param df: 包含 'market_mean_return' 列的 DataFrame
    :return: 增加 'market_mean_return_ma10_minus_ma5' 列的 DataFrame
    """
    df = df.copy()
    if 'market_mean_return' not in df.columns:
        df = calculate_market_mean_return(df)
    # 只保留每个日期一行
    macro = df[['date', 'market_mean_return']].drop_duplicates().sort_values('date')
    macro['ma10'] = macro['market_mean_return'].rolling(window=10, min_periods=8).mean()
    macro['ma5'] = macro['market_mean_return'].rolling(window=5, min_periods=4).mean()
    macro['market_mean_return_ma10_minus_ma5'] = macro['ma10'] - macro['ma5']
    df = df.merge(macro[['date', 'market_mean_return_ma10_minus_ma5']], on='date', how='left')
    return df

def calculate_market_mean_return_ma20_minus_ma10(df: pd.DataFrame):
    """
    计算 market_mean_return 的20日均线减去10日均线（全市场截面，所有股票每天该值一致）
    :param df: 包含 'market_mean_return' 列的 DataFrame
    :return: 增加 'market_mean_return_ma20_minus_ma10' 列的 DataFrame
    """
    df = df.copy()
    if 'market_mean_return' not in df.columns:
        df = calculate_market_mean_return(df)
    macro = df[['date', 'market_mean_return']].drop_duplicates().sort_values('date')
    macro['ma20'] = macro['market_mean_return'].rolling(window=20, min_periods=18).mean()
    macro['ma10'] = macro['market_mean_return'].rolling(window=10, min_periods=8).mean()
    macro['market_mean_return_ma20_minus_ma10'] = macro['ma20'] - macro['ma10']
    df = df.merge(macro[['date', 'market_mean_return_ma20_minus_ma10']], on='date', how='left')
    return df

@groupby_code
def calculate_volume_ma_ratio(df: pd.DataFrame):
    """
    计算成交量30日均线与10日均线的比值因子
    :param df: 包含 'volume' 列的 DataFrame
    :return: 增加 'volume_ma_ratio' 列的 DataFrame
    """
    df = df.copy()
    df['volume_ma30'] = df['volume'].rolling(window=30, min_periods=25).mean()
    df['volume_ma10'] = df['volume'].rolling(window=10, min_periods=8).mean()
    # 加上一个极小值避免除以0
    df['volume_ma_ratio'] = df['volume_ma30'] / (df['volume_ma10'] + 1e-10)
    return df

@groupby_code
def calculate_volume_ma_ratio2(df: pd.DataFrame):
    """
    计算成交量10日均线与5日均线的比值因子
    :param df: 包含 'volume' 列的 DataFrame
    :return: 增加 'volume_ma_ratio' 列的 DataFrame
    """
    df = df.copy()
    df['volume_ma30'] = df['volume'].rolling(window=30, min_periods=25).mean()
    # 加上一个极小值避免除以0
    df['volume_ma_ratio2'] = df['volume_ma30'] / (df['volume_ma10'] + 1e-10)
    return df

@groupby_code
def calculate_volume_ma_ratio3(df: pd.DataFrame):
    """
    计算成交量20日均线与5日均线的比值因子
    :param df: 包含 'volume' 列的 DataFrame
    :return: 增加 'volume_ma_ratio' 列的 DataFrame
    """
    df = df.copy()
    # 加上一个极小值避免除以0
    df['volume_ma_ratio3'] = df['volume_ma20'] / (df['volume_ma5'] + 1e-10)
    return df

@groupby_code
def calculate_volume_ma_min_pct(df: pd.DataFrame):
    """
    计算成交量过去30天的分位数排名（百分比）最小值越大排名越靠前
    :param df: 包含 'volume' 列的 DataFrame
    :return: 增加 'volume_ma_min_pct' 列的 DataFrame
    """
    df = df.copy()
    df['volume_ma_min_pct'] = np.log1p(df['volume']).rolling(30,min_periods=25).rank(pct=True, ascending=False)
    return df

@groupby_code
def calculate_chaikin_ad_line(df: pd.DataFrame, chaikin_window=10):
    """
    计算基于换手率的蔡金资金流(CMF)和A/D线(ADL)，并处理一字板情况。

    :param df: 包含'high', 'low', 'close', 'turn', 'pctChg'列的DataFrame
    :param chaikin_window: CMF的计算窗口，默认为20
    :return: 增加'chaikin_money_flow'和'ad_line'列的DataFrame
    """
    df = df.copy()
    high, low, close, turn, pct_change = df['high'], df['low'], df['close'], df['turn'], df['pctChg']

    # 1. 计算资金流乘数 (MFM)，处理一字板情况
    # 正常情况
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
    
    # 一字板情况
    is_one_word_board = (high == low)
    is_limit_up = (pct_change > 9.8)  # 涨停一字板
    is_limit_down = (pct_change < -9.8) # 跌停一字板
    
    # 根据一字板类型修正MFM
    mfm = np.where(is_one_word_board & is_limit_up, 0.5, mfm)
    mfm = np.where(is_one_word_board & is_limit_down, -0.5, mfm)
    mfm = np.where(is_one_word_board & ~is_limit_up & ~is_limit_down, 0.0, mfm) # 停牌等

    # 2. 计算资金流量 (MFV)，使用换手率
    money_flow_volume = mfm * turn

    df['ad_line'] = money_flow_volume.cumsum()
    return df

def calculate_factor_explosion_point(data: pd.DataFrame, 
                                     sentiment_threshold=90, 
                                     deviation_threshold=-5,
                                     momentum_threshold=20) -> pd.Series:
    """
    计算“三位一体起爆点因子” (Trinity Explosion Factor)。
    
    该因子融合了三个维度的信号，旨在寻找股价从极端超卖状态反转的“起爆点”。
    因子值为0-3的评分，分数越高，信号越强。
    设计上适用于预测未来短期（如5日）的反转收益。

    :param data: pd.DataFrame，必须包含 'high', 'low', 'close' 列
    :param sentiment_threshold: 散户线的情绪阈值
    :param deviation_threshold: 偏离度的负向偏离阈值
    :param momentum_threshold: 主力线的超卖区阈值
    :return: pd.Series，返回计算好的因子值
    """
    # --- 依赖计算 ---
    # 1. 计算散户线
    hhv_60 = data['high'].rolling(window=60, min_periods=55).max()
    llv_60 = data['low'].rolling(window=60, min_periods=55).min()
    denominator_sent = hhv_60 - llv_60
    retail_sentiment = 100 * (hhv_60 - data['close']) / denominator_sent
    retail_sentiment.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. 计算偏离度
    ma_4 = data['close'].rolling(window=4, min_periods=4).mean()
    deviation = (data['close'] - ma_4) / ma_4 * 100
    
    # 3. 计算主力线
    llv_30 = data['low'].rolling(window=30, min_periods=25).min()
    hhv_30 = data['high'].rolling(window=30, min_periods=25).max()
    relative_position = 100 * (data['close'] - llv_30) / (hhv_30 - llv_30)
    relative_position.replace([np.inf, -np.inf], np.nan, inplace=True)
    fast_smooth = relative_position.rolling(window=5, min_periods=3).mean()
    slow_smooth = fast_smooth.rolling(window=3, min_periods=2).mean()
    momentum_indicator = 3 * fast_smooth - 2 * slow_smooth
    smoothed_momentum = momentum_indicator.ewm(span=6, adjust=False).mean()

    # --- 因子评分计算 ---
    # 条件1：情绪冰点
    score1 = (retail_sentiment > sentiment_threshold).astype(int)
    
    # 条件2：动能衰竭 (负向偏离度极大，且开始回升)
    cond2 = (deviation < deviation_threshold) & (deviation > deviation.shift(1))
    score2 = cond2.astype(int)
    
    # 条件3：主力试探 (主力线在低位，且开始回升)
    cond3 = (smoothed_momentum < momentum_threshold) & (smoothed_momentum > smoothed_momentum.shift(1))
    score3 = cond3.astype(int)
    
    # 合成最终因子
    factor = score1 + score2 + score3
    data['explosion_point'] = factor
    return data

@groupby_code
def calculate_mmt_overnight_A(df: pd.DataFrame, window=5):
    """
    计算隔夜动量因子 mmt_overnight_A
    :param df: 包含 'open', 'close' 列的 DataFrame
    :param window: 动量回看窗口，默认为5
    :return: 增加 'mmt_overnight_A' 列的 DataFrame
    """
    df = df.copy()
    # 前一日收盘价
    close_t1 = df['close'].shift(1)
    # 隔夜收益率
    overnight_return = df['open'] / close_t1 - 1
    # N日移动平均隔夜动量
    df['mmt_overnight_A'] = overnight_return.rolling(window=window, min_periods=window-1).mean()
    return df

@groupby_code
def calculate_weighted_rsi(df: pd.DataFrame, window=10):
    """
    计算成交量加权RSI因子
    :param df: 包含 'close', 'volume' 列的 DataFrame
    :param window: RSI计算窗口，默认为14
    :return: 增加 'weighted_rsi' 列的 DataFrame
    """
    df = df.copy()
    # 计算涨跌幅
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # 计算成交量加权的涨跌幅
    vol = df['volume']
    weighted_gain = gain * vol
    weighted_loss = loss * vol
    # 计算N日加权均值
    avg_gain = pd.Series(weighted_gain).rolling(window=window, min_periods=window-2).mean()
    avg_loss = pd.Series(weighted_loss).rolling(window=window, min_periods=window-2).mean()
    # 计算加权RSI
    rs = avg_gain / (avg_loss + 1e-10)
    df['weighted_rsi'] = 100 / (1 + rs)
    return df

factor_names = ['volume_price_volatility',
                'amihud_illiquidity',
                'volume_ma_min_pct',
               ]

factor_dict = {
    'volume_price_divergence': mark_volume_price_divergence,
    'amihud_illiquidity': calculate_amihud_illiquidity,
    'volume_price_turn_body': calculate_volume_price_turnover_percentile,
    'volume_price_volatility': calculate_volume_price_volatility,
    'market_mean_return': calculate_market_mean_return,
    'market_mean_return_ma10_minus_ma5': calculate_market_mean_return_ma10_minus_ma5,
    'market_mean_return_ma20_minus_ma10': calculate_market_mean_return_ma20_minus_ma10,
    'volume_ma_ratio': calculate_volume_ma_ratio,
    'volume_ma_min_pct': calculate_volume_ma_min_pct,
    'ad_line': calculate_chaikin_ad_line,
    'explosion_point': calculate_factor_explosion_point,
    'mmt_overnight_A': calculate_mmt_overnight_A,
    'weighted_rsi': calculate_weighted_rsi,
    'volume_ma_ratio2': calculate_volume_ma_ratio2,
    'volume_ma_ratio3': calculate_volume_ma_ratio3,
}
# 标记是否为次数因子
mask_dict = {
    'volume_price_divergence': False,
    'amihud_illiquidity': False,
    'volume_price_turn_body': False,
    'volume_price_volatility': False,
    'market_mean_return': False,
    'market_mean_return_ma10_minus_ma5': False,
    'market_mean_return_ma20_minus_ma10': False,
    'volume_ma_ratio': False,
    'volume_ma_min_pct': False,
    'ad_line': False,
    'explosion_point': True,
    'weighted_rsi': False,
    'volume_ma_ratio2': False,
    'volume_ma_ratio3': False,
}

def get_factor_merge_table(factor_names=None):
    if not factor_names:
        factor_names = factor_dict.keys()
    df = get_stock_merge_industry_table(length=220)
    for name in factor_names:
        logging.info(f"计算因子: {name}")
        df = factor_dict[name](df)
    if 'industry' in df.columns:
        industry_counts = df['industry'].value_counts(dropna=False)
        logging.info(f"行业分布统计: {industry_counts.to_dict()}")
    else:
        logging.info("数据中未找到'industry'列，无法统计行业分布。")
    logging.info(df.head())
    logging.info(df.tail())
    logging.info(df.info())
    return df

"""
中性化因子相关性分析结果 (Top 10):
                  Factor_1                 Factor_2  Correlation
0   volume_price_turn_body  volume_price_volatility     0.775592
"""

if __name__ == "__main__":
    df = get_factor_merge_table(factor_names)