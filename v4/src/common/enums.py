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
    IS_COMPLETE = "is_complete"
    CALENDAR_DATE = "calendar_date"
    IS_TRADING_DAY = "is_trading_day"


FIELD_DTYPES = {
    Fields.DT: 'datetime64[ns]',
    Fields.T: 'object',
    Fields.OPEN: 'float64',
    Fields.HIGH: 'float64',
    Fields.LOW: 'float64',
    Fields.CLOSE: 'float64',
    Fields.VOL: 'int64',
    Fields.AMT: 'float64',
    Fields.SYMBOL: 'object',
    Fields.IS_COMPLETE: 'bool',
    Fields.CALENDAR_DATE: 'datetime64[ns]',
    Fields.IS_TRADING_DAY: 'bool',
}


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

    # Calendar Date
    'calendar_date': Fields.CALENDAR_DATE,
    'calendarDate': Fields.CALENDAR_DATE,

    # Is Trading Day
    'is_trading_day': Fields.IS_TRADING_DAY,
    'isTradingDay': Fields.IS_TRADING_DAY,
    'is_trade': Fields.IS_TRADING_DAY,
    'is_trad': Fields.IS_TRADING_DAY,
}

@unique
class DatabaseConfig(str, Enum):
    """
    数据库文件路径配置
    """
    # 统一的股票数据库
    STOCK_DB = 'd:/stock/v4/stock_data.db'

@unique
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
    # 股票数据状态
    STOCK_DATA_STATUS = 'stock_data_status'


# 定义索引策略
TABLE_INDEX_STRATEGY = {
    TableName.KLINE_5MIN.value: [Fields.SYMBOL.value, Fields.DT.value],
    TableName.STOCK_DATA_STATUS.value: [Fields.SYMBOL.value],
}