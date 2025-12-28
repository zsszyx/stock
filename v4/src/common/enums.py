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
    CALENDAR_DATE = "calendar_date"
    IS_TRADING_DAY = "is_trading_day"

    # Sync Status
    SYNC_START_DATE = "sync_start_date"
    SYNC_END_DATE = "sync_end_date"
    STATUS = "status"
    LAST_SYNC_TIME = "last_sync_time"


@unique
class SyncStatus(str, Enum):
    """
    数据同步状态
    """
    DATA_COMPLETE = "complete"
    DATA_INCOMPLETE = "incomplete"
    NO_DATA = "no_data"
    UP_TO_DATE = "up_to_date"


FIELD_DTYPES = {
    Fields.DT: 'TEXT',
    Fields.T: 'TEXT',
    Fields.OPEN: 'REAL',
    Fields.HIGH: 'REAL',
    Fields.LOW: 'REAL',
    Fields.CLOSE: 'REAL',
    Fields.VOL: 'REAL',
    Fields.AMT: 'REAL',
    Fields.SYMBOL: 'TEXT',
    Fields.CALENDAR_DATE: 'TEXT',
    Fields.IS_TRADING_DAY: 'INTEGER',
    Fields.SYNC_START_DATE: 'TEXT',
    Fields.SYNC_END_DATE: 'TEXT',
    Fields.STATUS: 'TEXT',
    Fields.LAST_SYNC_TIME: 'TEXT',
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
    # 同步状态表
    SYNC_STATUS = 'sync_status'


# 定义索引策略
TABLE_INDEX_STRATEGY = {
    TableName.KLINE_5MIN.value: [Fields.SYMBOL.value, Fields.DT.value],
    TableName.SYNC_STATUS.value: [Fields.SYMBOL.value],
}