import pandas as pd
import numpy as np
from .enums import Fields

# 定义标准的 DataFrame 数据类型 (Dtypes)
# 使用 numpy 类型以获得最佳性能
SCHEMA_DTYPES = {
    Fields.SYMBOL:    "string",      # 也就是 'string[python]'
    # datetime 通常设为 index, 不放在 column 里
    Fields.OPEN:      np.float64,
    Fields.HIGH:      np.float64,
    Fields.LOW:       np.float64,
    Fields.CLOSE:     np.float64,
    Fields.VOL:       np.float64,    # 用 float 存 volume 是为了兼容调整后的碎股
    Fields.AMT:       np.float64,
}

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    强制转换 DataFrame 为标准格式
    """
    # 强制类型转换 (astype copy=False 尽量避免内存拷贝)
    # 注意：Datetime 必须处理为标准 datetime64[ns]
    assert Fields.DT in df.columns, f'need {Fields.DT}'
    df[Fields.DT] = pd.to_datetime(df[Fields.DT])
    if Fields.T in df.columns:
        df[Fields.T] = pd.to_datetime(df[Fields.T])
    else:
        Warning
    

    return df.astype(SCHEMA_DTYPES, errors='ignore')