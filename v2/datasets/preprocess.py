import pandas as pd

def calc_ma30(df):
    """
    计算30日收盘价滑动均线和30日成交量均值，并去除前面没有均线的行
    :param df: 包含至少 'close' 和 'volume' 列的DataFrame
    :return: 新的DataFrame，包含'ma30'和'vol_ma30'列，且去除前面没有均线的行
    """
    df['ma30'] = df['close'].rolling(window=30).mean()
    df['vol_ma30'] = df['volume'].rolling(window=30).mean()
    df = df.dropna(subset=['ma30', 'vol_ma30'])
    return df
