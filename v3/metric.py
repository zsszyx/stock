import talib as ta

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