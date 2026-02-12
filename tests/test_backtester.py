import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock
from backtest.backtester import Backtester

def test_backtester_portfolio_transition():
    # Mock strategy
    mock_strategy = MagicMock()
    
    # Day 1 selection
    mock_strategy.select_top_stocks.side_effect = [
        ['S1', 'S2', 'S3'], # Day 1
        ['S1', 'S4', 'S5'], # Day 2
    ]
    
    # Mock daily context for prices
    # We need price data to calculate returns
    # Let's say every stock is $10 on day 1, and changes on day 2
    daily_data = pd.DataFrame([
        {'code': 'S1', 'date': '2025-01-01', 'close': 10.0, 'open': 10.0},
        {'code': 'S2', 'date': '2025-01-01', 'close': 10.0, 'open': 10.0},
        {'code': 'S3', 'date': '2025-01-01', 'close': 10.0, 'open': 10.0},
        {'code': 'S1', 'date': '2025-01-02', 'close': 11.0, 'open': 10.0, 'prev_close': 10.0},
        {'code': 'S2', 'date': '2025-01-02', 'close': 9.0,  'open': 10.0, 'prev_close': 10.0},
        {'code': 'S3', 'date': '2025-01-02', 'close': 10.0, 'open': 10.0, 'prev_close': 10.0},
        {'code': 'S4', 'date': '2025-01-02', 'close': 12.0, 'open': 10.0, 'prev_close': 10.0},
        {'code': 'S5', 'date': '2025-01-02', 'close': 8.0,  'open': 10.0, 'prev_close': 10.0},
    ])
    
    backtester = Backtester(initial_cash=100000, slots=3)
    
    # Run for two days
    # Day 1: Buys S1, S2, S3 at close of 2025-01-01 (simplified)
    # Day 2: 
    #   Calculate returns of S1, S2, S3 from Day 1 close to Day 2 close.
    #   S1: +10%, S2: -10%, S3: 0% -> Mean return: 0%
    #   Then Rebalance: Sell S2, S3. Keep S1. Buy S4, S5.
    
    # We need a way to provide prices to the backtester. 
    # Let's pass the daily_df to run.
    dates = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
    results = backtester.run(dates, mock_strategy, daily_data)
    
    assert len(results) == 2
    # Day 1: Portfolio [S1, S2, S3], value 100000
    # Day 2: Portfolio [S1, S4, S5], value (S1_new + S2_new + S3_new) = 11000 + 9000 + 10000 = 30000 (if each had 10000)
    # Wait, the math depends on how the backtester handles buying at close vs open.
    # Usually backtest: select at end of day T, buy at close of T or open of T+1.
    # Let's assume buy at close of T, sell at close of T+1.
    
    assert backtester.portfolio == {'S1', 'S4', 'S5'}
