from enum import Enum, IntEnum

class Exchange(Enum):
    """交易所标准定义"""
    SZSE = "SZ"  # 深交所
    SSE = "SH"   # 上交所
    BSE = "BJ"   # 北交所

class AdjustType(Enum):
    """复权类型"""
    NONE = "none"   # 不复权 (实盘用)
    PRE = "qfq"     # 前复权 (回测默认)
    POST = "hfq"    # 后复权
    
# 核心字段名映射（防止拼写错误）
class Fields:
    SYMBOL = "symbol"
    DT = "datetime"
    T = 'time'
    VOL = "volume"
    AMT = "amount"