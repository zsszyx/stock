import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime
from data_context.context import DailyContext

from selector.base_selector import BaseSelector

class KSPScoreSelector(BaseSelector):
    """
    Score Magnitude:
    1. If abs(PctChg) > 0.05: abs(Kurtosis) * abs(Skewness) * abs(PctChg)
    2. Otherwise: abs(Kurtosis) * abs(Skewness)
    
    Conditions for Negative Score (Penalty):
    1. Kurtosis < 0
    2. Skewness > 0
    3. abs(PctChg) > 0.05 (5%)
    4. POC < PrevDayClose * 0.98 (POC is > 2% lower than previous day's close)
    
    If any of the above is true, the score is negative.
    Otherwise, it is positive.
    """

    def get_scores(self, date: datetime, candidate_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate scores for all stocks on the given date.
        Returns a DataFrame with columns: ['code', 'score', 'kurt', 'skew', 'pct_chg']
        """
        # Get data for the specific date
        df = self.get_data(date, window_days=1, candidate_codes=candidate_codes)
        
        if df.empty:
            return pd.DataFrame()

        # Ensure columns exist
        required_cols = ['code', 'kurt', 'skew', 'pct_chg', 'poc', 'prev_close']
        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame()

        # Handle NaN values
        df = df.dropna(subset=required_cols).copy()
        
        # Calculate Base Magnitude: abs(Kurtosis) * abs(Skewness)
        base_magnitude = df['kurt'].abs() * df['skew'].abs()
        
        # PctChg is only multiplied as a penalty term when abs(PctChg) > 0.05
        magnitude = np.where(df['pct_chg'].abs() > 0.03, 
                             base_magnitude * df['pct_chg'].abs() * 100, 
                             base_magnitude)
        
        # Determine Sign
        # Negative if: 
        # 1. kurt < 0 
        # 2. skew > 0 
        # 3. abs(pct_chg) > 0.05
        # 4. poc < prev_close * 0.98 (POC is > 2% lower than previous day's close)
        is_bad = (df['kurt'] < 0) | \
                 (df['skew'] > 0) | \
                 (df['pct_chg'].abs() > 0.03) | \
                 (df['poc'] < df['prev_close'] * 0.98)
        
        # Apply sign
        # Where is_bad is True, score = -magnitude
        # Where is_bad is False, score = magnitude
        scores = np.where(is_bad, -magnitude, magnitude)
        
        result = df[['code', 'kurt', 'skew', 'pct_chg', 'poc']].copy()
        result['score'] = scores
        result = result.sort_values('score', ascending=False)
        
        return result

    def select(self, date: datetime, 
               candidate_codes: Optional[List[str]] = None, 
               top_n: int = 10) -> List[str]:
        """
        Select top N stocks with the highest KSP Score.
        """
        scores_df = self.get_scores(date, candidate_codes)
        if scores_df.empty:
            return []
            
        return scores_df.head(top_n)['code'].tolist()
