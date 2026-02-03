import numpy as np
from typing import List, Tuple, Optional, Union
from functools import cached_property

class DistributionAnalyzer:
    """
    Analyzes the distribution of Price and Volume data.
    
    Implements efficient, cached calculations for volume-weighted statistical moments
    and profile structure (POC).
    """
    
    
    def __init__(self, prices: Union[List[float], np.ndarray], volumes: Union[List[float], np.ndarray], poc_bins: int = 50):
        """
        Initialize the analyzer with raw data.
        Automatically filters out zero-volume data points.
        """
        # 1. Convert to Numpy Arrays for vectorization
        self._prices = np.asarray(prices, dtype=np.float64)
        self._volumes = np.asarray(volumes, dtype=np.float64)
        self._poc_bins = poc_bins
        
        # 2. Validation
        if self._prices.shape != self._volumes.shape:
            raise ValueError(f"Shape mismatch: prices {self._prices.shape} vs volumes {self._volumes.shape}")
            
        # 3. Pre-processing: Filter effective data (Volume > 0)
        mask = self._volumes > 0
        self.prices = self._prices[mask]
        self.volumes = self._volumes[mask]
        self.total_volume = np.sum(self.volumes) if len(self.volumes) > 0 else 0.0
        self.is_valid = len(self.prices) > 1 and self.total_volume > 0

    @cached_property
    def weighted_mean(self) -> float:
        """Calculates Volume-Weighted Average Price (VWAP)."""
        if not self.is_valid:
            return 0.0
        return np.average(self.prices, weights=self.volumes)

    @cached_property
    def weighted_std(self) -> float:
        """Calculates Volume-Weighted Standard Deviation."""
        if not self.is_valid:
            return 0.0
        variance = np.average((self.prices - self.weighted_mean)**2, weights=self.volumes)
        return np.sqrt(variance)

    @property
    def skewness(self) -> float:
        """
        Calculates Volume-Weighted Skewness (Fisher-Pearson coefficient).
        Negative Skew = P-shape (Tail on the left/low prices, Bulk on high prices).
        """
        if not self.is_valid or self.weighted_std == 0:
            return 0.0
            
        # S = Sum(w * (x - mean)^3) / (Sum(w) * std^3)
        skew_num = np.sum(self.volumes * (self.prices - self.weighted_mean)**3)
        skew_denom = self.total_volume * (self.weighted_std**3)
        
        return skew_num / skew_denom if skew_denom != 0 else 0.0

    @property
    def kurtosis(self) -> float:
        """
        Calculates Volume-Weighted Excess Kurtosis.
        High Kurtosis = Sharp peak, fat tails (Energy concentration).
        """
        if not self.is_valid or self.weighted_std == 0:
            return 0.0
            
        # K = Sum(w * (x - mean)^4) / (Sum(w) * std^4) - 3
        kurt_num = np.sum(self.volumes * (self.prices - self.weighted_mean)**4)
        kurt_denom = self.total_volume * (self.weighted_std**4)
        
        return (kurt_num / kurt_denom) - 3 if kurt_denom != 0 else 0.0

    @property
    def poc(self) -> float:
        """
        Calculates Point of Control (Price level with highest volume).
        Uses histogram binning defined by self._poc_bins.
        """
        if not self.is_valid:
            return self.prices[0] if len(self.prices) > 0 else 0.0
            
        min_p, max_p = np.min(self.prices), np.max(self.prices)
        if min_p == max_p:
            return min_p
            
        hist, bin_edges = np.histogram(self.prices, bins=self._poc_bins, weights=self.volumes, range=(min_p, max_p))
        max_idx = np.argmax(hist)
        # Return center of the bin
        return (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2

