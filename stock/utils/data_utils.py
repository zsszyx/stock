import pandas as pd
import numpy as np
from typing import List, Union

class DataUtils:
    """
    全项目统一的数据处理工具类 - 贯彻 DRY 原则
    """
    @staticmethod
    def normalize_code(code: str) -> str:
        """统一代码格式：sh.600000, sz.000001"""
        code = str(code).strip()
        if any('\u4e00' <= char <= '\u9fff' for char in code): return code
        if code.startswith(('sz.', 'sh.', 'bj.')): return code
        
        if len(code) == 6 and code.isdigit():
            if code.startswith(('600', '601', '603', '605', '688', '689')):
                return f'sh.{code}'
            return f'sz.{code}'
        return code

    @staticmethod
    def clean_numeric_df(df: pd.DataFrame, columns: List[str], fill_value: float = 0.0) -> pd.DataFrame:
        """统一数值转换与空值填充"""
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill_value)
        return df

    @staticmethod
    def align_to_index(df: pd.DataFrame, full_idx: pd.Index, sort_col: str = 'datetime') -> pd.DataFrame:
        """统一时间轴对齐逻辑 (ffill + bfill)"""
        if df.empty: return pd.DataFrame(index=full_idx)
        return df.set_index(sort_col).reindex(full_idx).ffill().bfill()
