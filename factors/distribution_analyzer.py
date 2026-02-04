import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

class DistributionAnalyzer:
    def __init__(self, prices, volumes, amounts=None, poc_bins=50):
        self.prices = np.array(prices, dtype=float)
        self.volumes = np.array(volumes, dtype=float)
        self.amounts = np.array(amounts, dtype=float) if amounts is not None else None
        self.poc_bins = poc_bins
        self._is_valid = len(self.prices) > 0 and np.sum(self.volumes) > 0

    @classmethod
    def from_amount_volume(cls, amounts, volumes, poc_bins=50):
        # Assuming price is amount / volume for each bar
        # Handle division by zero
        prices = np.divide(amounts, volumes, out=np.zeros_like(amounts, dtype=float), where=volumes!=0)
        return cls(prices, volumes, amounts=amounts, poc_bins=poc_bins)

    @property
    def real_price(self):
        if self.amounts is not None:
            return np.divide(self.amounts, self.volumes, out=np.zeros_like(self.amounts, dtype=float), where=self.volumes!=0)
        return self.prices

    @property
    def is_valid(self):
        return self._is_valid

    @property
    def skewness(self):
        if not self._is_valid:
            return 0.0
        # Weighted skewness
        prices = self.real_price
        mean = np.average(prices, weights=self.volumes)
        std = np.sqrt(np.average((prices - mean)**2, weights=self.volumes))
        if std == 0:
            return 0.0
        return np.average(((prices - mean) / std)**3, weights=self.volumes)

    @property
    def kurtosis(self):
        if not self._is_valid:
            return 0.0
        # Weighted kurtosis
        prices = self.real_price
        mean = np.average(prices, weights=self.volumes)
        std = np.sqrt(np.average((prices - mean)**2, weights=self.volumes))
        if std == 0:
            return 0.0
        return np.average(((prices - mean) / std)**4, weights=self.volumes) - 3.0

    @property
    def poc(self):
        if not self._is_valid:
            return 0.0
        
        prices = self.real_price
        min_price = np.min(prices)
        max_price = np.max(prices)
        
        if min_price == max_price:
            return min_price
            
        hist, bin_edges = np.histogram(
            prices, 
            bins=self.poc_bins, 
            weights=self.volumes,
            range=(min_price, max_price)
        )
        
        max_bin_idx = np.argmax(hist)
        poc_price = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
        return poc_price
