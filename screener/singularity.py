from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from stock.screener.base import BaseScreener
from stock.sql_op.op import SqlOp
from stock.sql_op import sql_config
from stock.factors.distribution_analyzer import DistributionAnalyzer

class SingularityScreener(BaseScreener):
    """
    Implements the Singularity Strategy Selection Logic.
    Filters: Skew < threshold
    Ranks: Kurtosis Descending
    """
    
    def __init__(self, skew_threshold: float = -0.0, top_n: int = 5):
        self.skew_threshold = skew_threshold
        self.top_n = top_n
        self.sql_op = SqlOp()
        
    def scan(self, date: datetime, codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scans for singularity candidates for the given date.
        
        Args:
            date: The date to scan.
            codes: Optional list of codes to restrict scan. If None, scans all active.
            
        Returns:
            DataFrame with columns: [code, skew, kurt, poc, close]
        """
        date_str = date.strftime('%Y-%m-%d')
        
        # 1. Determine Universe
        if codes is None:
            # Query all codes that have data on this date
            # Optimization: DISTINCT code from minutes where date = ...
            query = f"SELECT DISTINCT code FROM {sql_config.mintues5_table_name} WHERE date = '{date_str}'"
            res = self.sql_op.query(query)
            if res is None or res.empty:
                return pd.DataFrame()
            universe = res['code'].tolist()
        else:
            universe = codes
            
        # 2. Fetch Data & Calculate Factors
        results = []
        
        # We can batch query for performance if the universe is small, 
        # but for simplicity and safety against massive joins, we iterate or use a specialized query.
        # Let's try to fetch all data for this day in one go (Memory bound, but faster than loop).
        # Assuming < 5000 stocks * 48 bars = 240k rows. Pandas handles this easily.
        
        universe_str = "'" + "','".join(universe) + "'"
        data_query = f"""
            SELECT code, close, volume, amount
            FROM {sql_config.mintues5_table_name}
            WHERE date = '{date_str}'
        """
        # Note: If universe is partial, add AND code IN (...)
        if codes is not None:
             data_query += f" AND code IN ({universe_str})"
             
        df = self.sql_op.query(data_query)
        
        if df is None or df.empty:
            return pd.DataFrame()
            
        # Ensure numeric
        cols = ['close', 'volume', 'amount']
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        # Group by code and calculate
        grouped = df.groupby('code')
        
        for code, group in grouped:
            valid_group = group[group['volume'] > 0]
            if valid_group.empty:
                continue
                
            prices = (valid_group['amount'] / valid_group['volume']).tolist()
            volumes = valid_group['volume'].tolist()
            
            analyzer = DistributionAnalyzer(prices, volumes)
            if not analyzer.is_valid:
                continue
            
            poc = analyzer.poc
            skew = analyzer.skewness
            kurt = analyzer.kurtosis
            
            if skew is not None and kurt is not None:
                results.append({
                    'code': code,
                    'skew': skew,
                    'kurt': kurt,
                    'poc': poc,
                    'close': valid_group['close'].iloc[-1]
                })
                
        if not results:
            return pd.DataFrame()
            
        result_df = pd.DataFrame(results)
        
        # 3. Filter (Skew)
        filtered = result_df[result_df['skew'] < self.skew_threshold].copy()
        
        # 4. Rank (Kurt)
        ranked = filtered.sort_values(by='kurt', ascending=False)
        
        # 5. Return Top N (or all sorted if N is None)
        if self.top_n:
            return ranked.head(self.top_n)
        
        return ranked
