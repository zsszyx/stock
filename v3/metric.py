import talib as ta
import numpy as np
import pandas as pd

def groupby_code(func):
    """
    装饰器：自动在含有'code'列的大表上按股票分组批量计算。
    """
    def wrapper(df, *args, **kwargs):
        if 'code' in df.columns:
            if 'code' in df.index.names:
                df = df.reset_index(drop=True)
            return df.groupby('code').apply(lambda g: func(g, *args, **kwargs), include_groups=True)
        else:
            return func(df, *args, **kwargs)
    return wrapper

@groupby_code
def atr(df, timeperiod=14):
    """
    计算ATR（平均真实波幅）。
    """
    df['atr{}'.format(timeperiod)] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    return df

@groupby_code
def adtm(df, timeperiod=15):
    """
    计算ADTM（动态买卖气指标）。
    
    ADTM指标是一种判断市场买卖人气的指标，通过比较开盘价与最高价、最低价的关系来衡量买卖气势。
    
    参数:
    df: 包含股票数据的DataFrame，必须包含'open', 'high', 'low'列
    timeperiod: 计算周期，默认为23
    
    返回:
    添加了ADTM和ADTMMA列的DataFrame
    """
    # 计算DTM（买方动力）
    # 当日开盘价≥昨日开盘价时，DTM = max(当日最高价-当日开盘价, 当日开盘价-昨日开盘价)
    # 否则，DTM = 0
    df['open_prev'] = df['open'].shift(1)
    dtm = np.where(df['open'] >= df['open_prev'], 
                   np.maximum(df['high'] - df['open'], df['open'] - df['open_prev']), 
                   0)
    
    # 计算DBM（卖方动力）
    # 当日开盘价≤昨日开盘价时，DBM = max(当日开盘价-当日最低价, 当日开盘价-昨日开盘价)
    # 否则，DBM = 0
    dbm = np.where(df['open'] <= df['open_prev'], 
                   np.maximum(df['open'] - df['low'], df['open'] - df['open_prev']), 
                   0)
    
    # 计算STM（买方动力总和）
    stm = pd.Series(dtm, index=df.index).rolling(window=timeperiod).sum()
    
    # 计算SBM（卖方动力总和）
    sbm = pd.Series(dbm, index=df.index).rolling(window=timeperiod).sum()
    
    # 计算ADTM
    # 当STM > SBM时，ADTM = (STM - SBM) / STM
    # 当STM < SBM时，ADTM = (STM - SBM) / SBM
    # 当STM = SBM时，ADTM = 0
    adtm_values = np.where(stm > sbm, 
                           (stm - sbm) / stm, 
                           np.where(stm < sbm, 
                                    (stm - sbm) / sbm, 
                                    0))
    
    df[f'adtm_{timeperiod}'] = adtm_values / 100
    # 计算ADTM的移动平均
    df[f'adtmma_{timeperiod}'] = pd.Series(adtm_values, index=df.index).rolling(window=timeperiod).mean()
    
    # 删除中间计算列
    df.drop(columns=['open_prev'], inplace=True)
    
    return df


@groupby_code
def volume_ratio(df, timeperiod=5):
    """
    计算量比指标。
    
    量比是衡量相对成交量的指标，反映当前成交量与过去一段时间平均成交量的比值。
    
    参数:
    df: 包含股票数据的DataFrame，必须包含'volume'列
    timeperiod: 计算周期，默认为5
    
    返回:
    添加了量比列的DataFrame
    """
    # 计算过去timeperiod天的平均成交量（不包括当天）
    avg_volume = df['volume'].shift(1).rolling(window=timeperiod).mean()
    
    # 计算量比
    df[f'vr_{timeperiod}'] = df['volume'] / avg_volume / 10
    
    return df


@groupby_code
def vrsi(df, timeperiod=8):
    """
    计算VRSI（量相对强弱指数）。
    
    VRSI是基于成交量的相对强弱指数，用于分析成交量的变化趋势。
    
    参数:
    df: 包含股票数据的DataFrame，必须包含'volume'列
    timeperiod: 计算周期，默认为14
    
    返回:
    添加了VRSI列的DataFrame
    """
    # 计算成交量变化
    delta_volume = df['volume'].diff()
    
    # 分别计算上涨和下跌的成交量变化
    gain = delta_volume.where(delta_volume > 0, 0)
    loss = -delta_volume.where(delta_volume < 0, 0)
    
    # 计算平均上涨和下跌成交量（使用平滑移动平均）
    avg_gain = gain.rolling(window=timeperiod, min_periods=1).mean()
    avg_loss = loss.rolling(window=timeperiod, min_periods=1).mean()
    
    # 计算RS值（上涨平均成交量/下跌平均成交量）
    rs = avg_gain / avg_loss
    
    # 计算VRSI
    df[f'vrsi_{timeperiod}'] = (100 - (100 / (1 + rs)))/100
    
    return df


@groupby_code
def adx(df, timeperiod=14):
    """
    计算ADX（平均趋向指数）。
    
    ADX用于衡量趋势的强度，不指示趋势方向。
    
    参数:
    df: 包含股票数据的DataFrame，必须包含'high', 'low', 'close'列
    timeperiod: 计算周期，默认为14
    
    返回:
    添加了ADX列的DataFrame
    """
    df[f'adx_{timeperiod}'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=timeperiod) / 100
    df[f'adxr_{timeperiod}'] = ta.ADXR(df['high'], df['low'], df['close'], timeperiod=timeperiod) / 100
    return df

@groupby_code
def price_change(df, timeperiod=1):
    """
    计算价格变化率。
    
    参数:
    df: 包含股票数据的DataFrame，必须包含'close'列
    timeperiod: 计算周期，默认为1
    
    返回:
    添加了价格变化率列的DataFrame
    """
    df['open_pct'] = df['open'].pct_change(periods=timeperiod)
    df['high_pct'] = df['high'].pct_change(periods=timeperiod)
    df['low_pct'] = df['low'].pct_change(periods=timeperiod)
    df['close_pct'] = df['close'].pct_change(periods=timeperiod)
    return df

@groupby_code
def rsi(df, timeperiod=14):
    """
    计算RSI（相对强弱指数）。
    
    RSI用于衡量价格动量，不指示趋势方向。
    
    参数:
    df: 包含股票数据的DataFrame，必须包含'close'列
    timeperiod: 计算周期，默认为14
    
    返回:
    添加了RSI列的DataFrame
    """
    df[f'rsi_{timeperiod}'] = ta.RSI(df['close'], timeperiod=timeperiod) / 100
    return df