from typing import List, Dict
import numpy as np
import pandas as pd
from datetime import datetime
from stock.backtest.strategy import Strategy
from stock.backtest.models import OrderType

class SkewKurtStrategy(Strategy):
from typing import List, Dict
import numpy as np
import pandas as pd
from datetime import datetime
from stock.backtest.strategy import Strategy
from stock.backtest.models import OrderType

class SkewKurtStrategy(Strategy):
    def initialize(self):
        print("Initializing Skewness-Kurtosis Strategy (VWAP Based)...")
        self.window_days = 5  # Analyze last 5 days
        self.bars_per_day = 48 # Assuming 5-min bars (4 hours)
        self.window_size = self.window_days * self.bars_per_day
        self.rebalance_interval = 1 # Rebalance every day
        self.top_n = 5 # Hold top 5 stocks
        self.last_rebalance_date = None

    def _filter_stock(self, code: str) -> bool:
        """
        Filter for Main Board and ChiNext only.
        Exclude Indices, STAR Market, BSE.
        """
        if "." in code:
            prefix, num_code = code.split(".")
            if prefix == 'sh' and num_code.startswith('000'): return False
            if prefix == 'sz' and num_code.startswith('399'): return False
        else:
            num_code = code

        if num_code.startswith(('60', '00', '30')):
            return True
        
        return False

    def _weighted_moments(self, values, weights):
        """
        Computes weighted mean, std, skewness, and kurtosis.
        """
        if len(values) == 0 or weights.sum() == 0:
            return 0, 0, 0, 0
            
        mean = np.average(values, weights=weights)
        
        diff = values - mean
        
        variance = np.average(diff**2, weights=weights)
        std_dev = np.sqrt(variance)
        
        if std_dev == 0:
            return mean, 0, 0, 0
            
        m3 = np.average(diff**3, weights=weights)
        skew = m3 / (std_dev**3)
        
        m4 = np.average(diff**4, weights=weights)
        kurt = m4 / (std_dev**4) - 3
        
        return mean, std_dev, skew, kurt

    def next(self, bars):
        current_time = list(bars.values())[0].time
        current_date = current_time.date()
        
        # Only rebalance and make decisions if it's a new day and near market close
        if self.last_rebalance_date == current_date:
            return
            
        if current_time.hour < 14 or (current_time.hour == 14 and current_time.minute < 50):
            return

        self.last_rebalance_date = current_date
        print(f"Analyzing for rebalance: {current_date}")

        # 1. Calculate Statistics for all stocks
        scores = []
        
        for code in self.data_history:
            if code not in bars:
                continue

            if not self._filter_stock(code):
                continue
            
            history = self.data_history[code]
            if len(history) < self.window_size:
                continue
            
            # Get window
            window_bars = history[-self.window_size:]
            
            closes = np.array([b.close for b in window_bars])
            volumes = np.array([b.volume for b in window_bars])
            amounts = np.array([b.amount for b in window_bars])
            
            # Calculate VWAP prices for moments calculation
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                prices = np.where(volumes > 0, amounts / volumes, closes)
            
            # Filter 1: Exclude if price increased > 5% in the window
            if closes[0] > 0:
                window_return = (closes[-1] - closes[0]) / closes[0]
            else:
                window_return = 0.0
                
            if window_return > 0.05:
                continue
            
            if volumes.sum() == 0:
                continue
                
            # Use VWAP prices for moments calculation as per reference strategy
            mean, std, skew, kurt = self._weighted_moments(prices, volumes)
            
            # Filter 2: Negative Skewness
            if skew >= 0:
                continue

            # Filter 3: Current Day Down State (Close < Open)
            # Get the first bar of the current day for day_open_price
            day_history_current_date = [b for b in history if b.time.date() == current_date]
            if not day_history_current_date: 
                continue
            day_open_price = day_history_current_date[0].open
            
            current_close_price = bars[code].close 

            if current_close_price >= day_open_price:
                continue # Not in a 'down state' today

            scores.append({
                'code': code, 
                'kurt': kurt, 
                'skew': skew
            })

        # 2. Select Top N
        if not scores:
            print("No stocks passed all filters for this rebalance.")
            top_picks = [] 
        else:
            scores.sort(key=lambda x: x['kurt'], reverse=True)
            top_picks_data = scores[:self.top_n]
            top_picks = [x['code'] for x in top_picks_data]
            
            print(f"Top {self.top_n} Picks (VWAP Skew < 0 & Down Today -> High Kurt): {[(x['code'], f'K:{x['kurt']:.2f}', f'S:{x['skew']:.2f}') for x in top_picks_data]}")

        # 3. Rebalance Portfolio
        current_positions = list(self.broker.positions.keys())
        
        # Sell
        for code in current_positions:
            pos = self.broker.get_position(code)
            if pos.quantity > 0 and code not in top_picks:
                self.sell(code, pos.quantity)
                
        # Buy
        # Estimate total equity
        total_value = self.broker.get_portfolio_value({c: bars[c].close for c in bars if c in bars})
        target_allocation = 1.0 / self.top_n if self.top_n > 0 else 0
        target_value_per_stock = total_value * target_allocation
        
        for code in top_picks:
            if code not in bars: continue
            
            price = bars[code].close
            pos = self.broker.get_position(code)
            current_holding_value = pos.quantity * price
            
            diff_value = target_value_per_stock - current_holding_value
            
            if diff_value > price * 100:
                qty = int(diff_value / price / 100) * 100
                if qty > 0:
                    self.buy(code, qty)