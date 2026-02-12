import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Set

class Backtester:
    def __init__(self, initial_cash: float = 100000.0, slots: int = 9):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.slots = slots
        self.portfolio: Set[str] = set()
        self.history = []

    def run(self, dates: List[datetime], strategy, daily_df: pd.DataFrame):
        """
        Run backtest over a list of dates.
        daily_df must contain ['code', 'date', 'close', 'prev_close']
        """
        # Ensure date format in daily_df matches string representation of dates
        daily_df = daily_df.copy()
        if not isinstance(daily_df['date'].iloc[0], str):
            daily_df['date'] = daily_df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x))

        current_value = self.initial_cash
        
        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')
            
            # 1. Calculate returns of current portfolio for today
            today_value = self.cash
            if self.portfolio:
                # Get today's close prices for held stocks
                today_prices = daily_df[(daily_df['date'] == date_str) & (daily_df['code'].isin(self.portfolio))]
                
                # We assume equal weight from yesterday's close
                # Return = Sum(Weight_i * (Close_i / PrevClose_i))
                # where Weight_i = (TotalValue / Slots) / TotalValue = 1 / Slots
                
                held_count = len(self.portfolio)
                slot_value = (current_value - self.cash) / held_count if held_count > 0 else 0
                
                for _, row in today_prices.iterrows():
                    # If we don't have prev_close, we assume no change for that stock (simplified)
                    pct_chg = row.get('pct_chg', 0)
                    if pd.isna(pct_chg) and 'prev_close' in row and row['prev_close'] > 0:
                        pct_chg = (row['close'] - row['prev_close']) / row['prev_close']
                    elif pd.isna(pct_chg):
                        pct_chg = 0
                        
                    today_value += slot_value * (1 + pct_chg)
                
                # If some held stocks are missing from today's data (e.g. suspended)
                missing_count = held_count - len(today_prices)
                if missing_count > 0:
                    today_value += slot_value * missing_count

            current_value = today_value
            
            # 2. Rebalance: Get selection for today (to be held from today's close onwards)
            target_stocks = strategy.select_top_stocks(date)
            target_stocks = target_stocks[:self.slots] # Ensure limit
            
            # FIFO-like transition:
            # - Keep what is still in target
            # - Sell what is not in target
            # - Buy new targets in order of ranking to fill slots
            
            new_portfolio = set()
            # Keep existing ones that are in target
            for stock in self.portfolio:
                if stock in target_stocks:
                    new_portfolio.add(stock)
            
            # Fill remaining slots with new ones from target
            for stock in target_stocks:
                if len(new_portfolio) >= self.slots:
                    break
                new_portfolio.add(stock)
            
            self.portfolio = new_portfolio
            # In this equal-weight model, we assume we rebalance cash to be 0 
            # and fully invest the current_value into the new_portfolio slots.
            # (Simplified: ignoring transaction costs for now unless specified)
            self.cash = 0 if self.portfolio else current_value
            
            self.history.append({
                'date': date_str,
                'total_value': current_value,
                'portfolio': list(self.portfolio),
                'return': (current_value / self.initial_cash) - 1
            })
            
        return self.history
