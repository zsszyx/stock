import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from functools import cached_property

class DistributionAnalyzer:
    def __init__(self, prices, volumes, amounts=None, times=None, poc_bins=50):
        self.prices = np.array(prices, dtype=float)
        self.volumes = np.array(volumes, dtype=float)
        self.amounts = np.array(amounts, dtype=float) if amounts is not None else None
        self.times = np.array(times, dtype=str) if times is not None else None
        self.poc_bins = poc_bins
        self._is_valid = len(self.prices) > 0 and np.sum(self.volumes) > 0

    @classmethod
    def from_amount_volume(cls, amounts, volumes, times=None, poc_bins=50):
        # Assuming price is amount / volume for each bar
        # Handle division by zero
        prices = np.divide(amounts, volumes, out=np.zeros_like(amounts, dtype=float), where=volumes!=0)
        return cls(prices, volumes, amounts=amounts, times=times, poc_bins=poc_bins)

    @property
    def real_price(self):
        if self.amounts is not None:
            return np.divide(self.amounts, self.volumes, out=np.zeros_like(self.amounts, dtype=float), where=self.volumes!=0)
        return self.prices

    @property
    def morning_mean(self):
        if not self._is_valid or self.times is None:
            return 0.0
        # Time format is YYYYMMDDHHMMSSmmm, HH starts at index 8
        mask = np.array([t[8:10] < '12' for t in self.times])
        m_prices = self.real_price[mask]
        m_vols = self.volumes[mask]
        if len(m_prices) == 0 or np.sum(m_vols) == 0:
            return 0.0
        return np.average(m_prices, weights=m_vols)

    @property
    def afternoon_mean(self):
        if not self._is_valid or self.times is None:
            return 0.0
        # Time format is YYYYMMDDHHMMSSmmm, HH starts at index 8
        mask = np.array([t[8:10] >= '12' for t in self.times])
        a_prices = self.real_price[mask]
        a_vols = self.volumes[mask]
        if len(a_prices) == 0 or np.sum(a_vols) == 0:
            return 0.0
        return np.average(a_prices, weights=a_vols)

    @property
    def open(self):
        if not self._is_valid:
            return 0.0
        return self.real_price[0]

    @property
    def high(self):
        if not self._is_valid:
            return 0.0
        return np.max(self.real_price)

    @property
    def low(self):
        if not self._is_valid:
            return 0.0
        return np.min(self.real_price)

    @property
    def min_time(self):
        if not self._is_valid or self.times is None:
            return ""
        return self.times[np.argmin(self.real_price)]

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
