from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, time

# Adjust import based on execution context
try:
    from stock.backtest.strategy import Strategy
    from stock.backtest.models import Bar, Direction
    from stock.sql_op.op import SqlOp
    from stock.sql_op import sql_config
    from stock.screener.singularity import SingularityScreener
except ImportError:
    # Fallback if running from inside the package or different path
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from stock.backtest.strategy import Strategy
    from stock.backtest.models import Bar, Direction
    from stock.sql_op.op import SqlOp
    from stock.sql_op import sql_config
    from stock.screener.singularity import SingularityScreener

class SingularityStrategy(Strategy):
    """
    Singularity Breakout Strategy
    
    Logic:
    1. Daily Full Market Scan (Cross-Sectional):
       - Uses SingularityScreener (Skew < 0, High Kurtosis)
    2. Trigger: Close > POC
    """
    
    def __init__(self, broker, skew_threshold: float = -0.0, kurt_threshold: float = 0.0, top_n: int = 5):
        super().__init__(broker)
        self.top_n = top_n
        # Initialize the Screener Component
        self.screener = SingularityScreener(skew_threshold=skew_threshold, top_n=top_n)
        
        self.intraday_data: Dict[str, List[Bar]] = {}
        
    def initialize(self):
        print("SingularityStrategy Initialized (Delegating selection to SingularityScreener).")

    def next(self, bars: Dict[str, Bar]):
        # 1. Accumulate Data
        for code, bar in bars.items():
            if code not in self.intraday_data:
                self.intraday_data[code] = []
            self.intraday_data[code].append(bar)
            
        # 2. Check for Market Close (15:00)
        if not bars:
            return
            
        current_time = list(bars.values())[0].time
        if current_time.time() == time(15, 0):
            self._rebalance_portfolio(current_time, bars)
            # Clear for next day
            self.intraday_data = {}

    def _rebalance_portfolio(self, current_time, current_bars):
        # Delegate selection to the Screener
        # We pass the universe of codes we are currently tracking/simulating
        # In a full market simulation, we might pass None to scan everything in DB.
        # Here, to align with the backtest engine's scope, we scan the codes present in intraday_data.
        universe = list(self.intraday_data.keys())
        
        # The screener returns the Top N DataFrame
        top_picks = self.screener.scan(current_time, codes=universe)
        
        if top_picks.empty:
            return

        # 3. Generate Target Portfolio
        target_codes = []
        for _, row in top_picks.iterrows():
            if row['close'] > row['poc']:
                target_codes.append(row['code'])
                
        # 4. Execute Trades
        current_positions = [code for code, pos in self.broker.positions.items() if pos.quantity > 0]
        
        # Sell Logic
        for code in current_positions:
            if code not in target_codes:
                pos = self.broker.get_position(code)
                print(f"[{current_time}] SELL {code} (No longer in Top {self.top_n} or Signal lost)")
                price = current_bars.get(code, self.intraday_data[code][-1]).close
                self.sell(code, pos.quantity, price)
        
        # Buy Logic
        held_count = len([c for c in current_positions if c in target_codes])
        empty_slots = self.top_n - held_count
        
        if empty_slots > 0:
            new_buys = [c for c in target_codes if c not in current_positions]
            current_cash = self.broker.cash
            if new_buys and current_cash > 0:
                amount_per_trade = current_cash / len(new_buys)
                
                for code in new_buys:
                    price = current_bars.get(code, self.intraday_data[code][-1]).close
                    qty = int(amount_per_trade // price)
                    qty = (qty // 100) * 100
                    
                    if qty > 0:
                        row = top_picks[top_picks['code'] == code].iloc[0]
                        print(f"[{current_time}] BUY {code} @ {price:.2f} (S:{row['skew']:.2f}, K:{row['kurt']:.2f}, POC:{row['poc']:.2f})")
                        self.buy(code, qty, price)
