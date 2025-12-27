from enum import Enum, unique


@unique
class Fields(str, Enum):
    """
    DataFrame的列名
    """
    DT = "datetime"
    T = "Time"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOL = "volume"
    AMT = "amount"
    SYMBOL = "symbol"


# Mapping from common aliases to the standard Fields enum
COLUMN_ALIASES = {
    # Datetime
    'datetime': Fields.DT,
    'DateTime': Fields.DT,
    'Date': Fields.DT,
    'date': Fields.DT,
    '日期': Fields.DT,

    # Time
    't': Fields.T,
    'time': Fields.T,
    'Time': Fields.T,
    '时间': Fields.T,
    '分钟': Fields.T,


    # Symbol
    'symbol': Fields.SYMBOL,
    'Symbol': Fields.SYMBOL,
    'code': Fields.SYMBOL,
    '股票代码': Fields.SYMBOL,
    '代码': Fields.SYMBOL,

    # OHLC
    'open': Fields.OPEN,
    'Open': Fields.OPEN,
    'high': Fields.HIGH,
    'High': Fields.HIGH,
    'low': Fields.LOW,
    'Low': Fields.LOW,
    'close': Fields.CLOSE,
    'Close': Fields.CLOSE,
    'O': Fields.OPEN,
    'H': Fields.HIGH,
    'L': Fields.LOW,
    'C': Fields.CLOSE,
    'o': Fields.OPEN,
    'h': Fields.HIGH,
    'l': Fields.LOW,
    'c': Fields.CLOSE,
    '开': Fields.OPEN,
    '高': Fields.HIGH,
    '低': Fields.LOW,
    '收': Fields.CLOSE,
    '开盘价': Fields.OPEN,
    '最高价': Fields.HIGH,
    '最低价': Fields.LOW,
    '收盘价': Fields.CLOSE,
    '开盘': Fields.OPEN,
    '收盘': Fields.CLOSE,
    '最高': Fields.HIGH,
    '最低': Fields.LOW,

    # Volume
    'volume': Fields.VOL,
    'Volume': Fields.VOL,
    'vol': Fields.VOL,
    'Vol': Fields.VOL,
    '成交量': Fields.VOL,
    '量': Fields.VOL,

    # Amount
    'amount': Fields.AMT,
    'Amount': Fields.AMT,
    'Amo': Fields.AMT,
    'amo': Fields.AMT,
    '成交额': Fields.AMT,
    '额': Fields.AMT,
}

class DatabaseConfig(str, Enum):
    """
    数据库文件路径配置
    """
    # 元数据数据库 (股票列表、交易日等)
    META = 'd:/stock/v4/baostock_meta.db'
    # K线行情数据库
    KLINE = 'd:/stock/v4/kline_data.db'

class TableName(str, Enum):
    """
    数据表名枚举
    """
    # 股票列表
    STOCK_LIST = 'stock_list'
    # 交易日历
    TRADE_DATES = 'trade_dates'
    # 5分钟K线数据
    KLINE_5MIN = 'kline_5min'
    # K线数据范围记录
    KLINE_DATA_RANGE = 'kline_data_range'