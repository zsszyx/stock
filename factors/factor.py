import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_prepare.prepare import get_stock_merge_industry_table
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

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
    close_ma = df['close'].rolling(window=20).mean()
    volume_ma = df['volume'].rolling(window=20).mean()
    norm_volume = (df['volume'] - volume_ma)/volume_ma
    norm_close = (df['close'] - close_ma)/close_ma
    df['volume_price_divergence'] = abs(norm_close)/abs(norm_volume)
    df['volume_price_divergence'] = np.log1p(df['volume_price_divergence'])
    df['volume_price_divergence'] = np.where(norm_close < 0, df['volume_price_divergence'], -df['volume_price_divergence'])
    df['volume_price_divergence'] = pd.Series(df['volume_price_divergence']).rolling(window=10).mean()
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
    amihud_factor = daily_illiquidity.rolling(window=window).mean()
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
    turn_percentile = 1/df['turn'].rolling(window=window).rank(pct=True)
    turn_percentile = np.log1p(turn_percentile)
    volume_percentile = 1/df['volume'].rolling(window=window).rank(pct=True)
    volume_percentile = np.log1p(volume_percentile)
    body_percentile = 1/abs(df['close'] - df['open']).rolling(window=window).rank(pct=True)
    body_percentile = np.log1p(body_percentile)
    df['volume_price_turn_body'] = body_percentile * volume_percentile
    df['volume_price_turn_body'] = df['volume_price_turn_body'].rolling(window=10).mean()
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
    volume_percent = df['volume'].rolling(window=window).rank(pct=True, ascending=False)
    volatility = df['close'].rolling(window=window).std()/df['close'].rolling(window=window).mean()
    volatility = np.log1p(1/volatility)
    df['volume_price_volatility'] = volume_percent * volatility
    df['volume_price_volatility'] = df['volume_price_volatility'].rolling(window=10).mean()
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


factor_names = ['volume_price_volatility', 'market_mean_return']

factor_dict = {
    'volume_price_divergence': mark_volume_price_divergence,
    'amihud_illiquidity': calculate_amihud_illiquidity,
    'volume_price_turn_body': calculate_volume_price_turnover_percentile,
    'volume_price_volatility': calculate_volume_price_volatility,
    'market_mean_return': calculate_market_mean_return,
}
# 标记是否为次数因子
mask_dict = {
    'volume_price_divergence': False,
    'amihud_illiquidity': False,
    'volume_price_turn_body': False,
    'volume_price_volatility': False,
    'market_mean_return': False,
}

def get_factor_merge_table(factor_names=None):
    if not factor_names:
        factor_names = factor_dict.keys()
    df = get_stock_merge_industry_table(length=200)
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


if __name__ == "__main__":
    df = get_factor_merge_table(factor_names)