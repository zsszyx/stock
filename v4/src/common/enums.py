from enum import Enum


class Fields(str, Enum):
    """
    DataFrame的列名
    """
    DT = "datetime"
    T = 'Time'
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOL = "volume"
    AMT = "amount"
    SYMBOL = "symbol"