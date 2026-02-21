"""
数据加载器模块 - 回测数据管理

功能:
- 从数据库加载股票日线数据
- 验证数据有效性
- 过滤无效股票 (零成交、NaN等)
- 数据对齐处理
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from stock.config import settings


class DataLoader:
    """股票数据加载器"""
    
    def __init__(self, repo=None):
        self.repo = repo
        self.cache = {}
    
    def load_daily_data(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载日线数据
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        
        Returns:
            DataFrame with columns: code, date, open, high, low, close, volume, amount
        """
        if not self.repo:
            raise ValueError("Repository not initialized")
        
        table_name = settings.TABLE_DAILY
        
        # 查询数据
        query = f"""
            SELECT code, date, open, high, low, close, volume, amount
            FROM {table_name}
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            AND code IN ({','.join([f"'{c}'" for c in codes])})
            ORDER BY code, date
        """
        df = self.repo.query(query)
        
        if df.empty:
            return df
        
        # 数据类型转换
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def validate_stock_data(self, df: pd.DataFrame) -> bool:
        """
        验证单只股票数据是否有效
        
        Args:
            df: 单只股票的数据
        
        Returns:
            True if valid, False otherwise
        """
        if df.empty:
            return False
        
        # 检查成交量
        if (df['volume'] <= 0).any():
            return False
        
        # 检查价格
        if df['close'].isna().any():
            return False
        
        return True
    
    def filter_valid_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤掉无效股票
        
        Args:
            df: 包含多只股票的数据
        
        Returns:
            过滤后的数据
        """
        if df.empty:
            return df
        
        # 按股票分组检查有效性
        valid_codes = []
        for code in df['code'].unique():
            stock_df = df[df['code'] == code]
            if self.validate_stock_data(stock_df):
                valid_codes.append(code)
        
        return df[df['code'].isin(valid_codes)]
    
    def align_data(self, df: pd.DataFrame, dates: List[str]) -> pd.DataFrame:
        """
        对齐数据到指定日期
        
        Args:
            df: 股票数据
            dates: 需要对齐的日期列表
        
        Returns:
            对齐后的数据
        """
        if df.empty:
            return df
        
        # 创建完整的日期索引
        date_index = pd.to_datetime(dates)
        
        # 按股票处理
        aligned_dfs = []
        for code in df['code'].unique():
            stock_df = df[df['code'] == code].copy()
            stock_df = stock_df.set_index('date')
            
            # 重新索引并填充
            stock_df = stock_df.reindex(date_index)
            stock_df['code'] = code
            stock_df = stock_df.ffill().bfill()
            stock_df = stock_df.reset_index().rename(columns={'index': 'date'})
            
            aligned_dfs.append(stock_df)
        
        if aligned_dfs:
            return pd.concat(aligned_dfs, ignore_index=True)
        return df


class StockDataValidator:
    """股票数据验证器"""
    
    @staticmethod
    def check_missing_data(df: pd.DataFrame) -> Dict[str, int]:
        """检查缺失数据"""
        return df.isnull().sum().to_dict()
    
    @staticmethod
    def check_zero_volume(df: pd.DataFrame) -> int:
        """检查零成交"""
        return (df['volume'] == 0).sum()
    
    @staticmethod
    def get_data_quality_report(df: pd.DataFrame) -> Dict:
        """生成数据质量报告"""
        return {
            'total_rows': len(df),
            'total_stocks': df['code'].nunique(),
            'date_range': f"{df['date'].min()} ~ {df['date'].max()}",
            'missing_data': StockDataValidator.check_missing_data(df),
            'zero_volume_days': StockDataValidator.check_zero_volume(df),
        }
