import pytest
from unittest.mock import Mock
from datetime import datetime, time, date
import pandas as pd
from stock.backtest.models import Bar

# Revert to absolute imports for easier pytest discovery with PYTHONPATH
from stock.backtest.strategies.singularity_backtest_strategy import SingularityStrategy

# Helper function to create a mock Bar object
def create_mock_bar(code, current_datetime, open_price, close_price):
    mock_bar = Mock(spec=Bar)
    mock_bar.code = code
    mock_bar.time = current_datetime
    mock_bar.date = current_datetime.date()
    mock_bar.open = open_price
    mock_bar.close = close_price
    mock_bar.high = max(open_price, close_price)
    mock_bar.low = min(open_price, close_price)
    mock_bar.volume = 1000
    mock_bar.amount = mock_bar.volume * (open_price + close_price) / 2
    return mock_bar

@pytest.fixture
def mock_broker():
    broker = Mock()
    broker.cash = 100000.0
    broker.positions = {} # Mock empty positions for simplicity
    broker.get_position.return_value = Mock(quantity=0) # No existing positions
    # Mock the buy method to record calls
    broker.buy = Mock(return_value=True) 
    broker.sell = Mock(return_value=True) # Mock sell too
    return broker

@pytest.fixture
def mock_screener():
    screener = Mock()
    return screener

@pytest.fixture

def mock_context():

    context = Mock()

    context.get_trading_dates.return_value = ['2026-02-04', '2026-02-05']

    return context



@pytest.fixture

def strategy(mock_broker, mock_screener, mock_context):

    # 初始化策略，注入 context

    s = SingularityStrategy(mock_broker, mock_context)

    # Mock 流水线

    s.pipeline = Mock()

    # 模拟 pipeline.run(current_time, context=self.context)

    def mock_run(date, context=None):

        return mock_screener.scan(context)

    

    s.pipeline.run = mock_run

    s.initialize()

    return s



def test_buy_within_2_percent_of_prev_close(strategy, mock_broker, mock_screener):

    test_code = 'test_stock_001'

    day1_date = datetime(2026, 2, 4, 15, 0, 0)

    day1_close_price = 100.0

    day1_poc = 100.0

    

    # Mock pipeline.run 返回结果

    mock_screener.scan.return_value = pd.DataFrame([{

        'code': test_code, 

        'skew': -0.1, 

        'kurt': 3.0, 

        'poc': day1_poc, 

        'close': day1_close_price

    }])

    

    # Simulate Day 1 close

    day1_bars = {test_code: create_mock_bar(test_code, day1_date, 99.0, day1_close_price)}

    strategy.next(day1_bars)

    

    assert len(strategy.pending_buys) == 1

    

    # Simulate Day 2 open

    day2_date = datetime(2026, 2, 5, 9, 35, 0)

    day2_open_price = 101.0 

    day2_bars = {test_code: create_mock_bar(test_code, day2_date, day2_open_price, 101.5)}

    strategy.next(day2_bars)

    

    assert mock_broker.submit_order.called

    assert len(strategy.pending_buys) == 0



def test_no_buy_above_2_percent_of_prev_close(strategy, mock_broker, mock_screener):

    test_code = 'test_stock_002'

    day1_date = datetime(2026, 2, 4, 15, 0, 0)

    day1_close_price = 100.0

    

    mock_screener.scan.return_value = pd.DataFrame([{

        'code': test_code, 

        'skew': -0.1, 

        'kurt': 3.0, 

        'poc': 100.0, 

        'close': day1_close_price

    }])

    

    day1_bars = {test_code: create_mock_bar(test_code, day1_date, 99.0, day1_close_price)}

    strategy.next(day1_bars)

    

    day2_date = datetime(2026, 2, 5, 9, 35, 0)

    day2_open_price = 102.5 # > 2%

    day2_bars = {test_code: create_mock_bar(test_code, day2_date, day2_open_price, 103.0)}

    strategy.next(day2_bars)

    

    assert not mock_broker.submit_order.called

    assert len(strategy.pending_buys) == 0



def test_fifo_rotation(strategy, mock_broker, mock_screener):

    initial_holdings = ['s1', 's2', 's3', 's4', 's5']

    strategy.holdings_order = initial_holdings.copy()

    mock_broker.get_position.side_effect = lambda code: Mock(quantity=1000) if code in initial_holdings else Mock(quantity=0)

    

    current_time = datetime(2026, 2, 4, 15, 0, 0)

    test_code_new = 'test_new_001'

    

    mock_screener.scan.return_value = pd.DataFrame([{

        'code': test_code_new, 

        'skew': -0.1, 

        'kurt': 3.0, 

        'poc': 100.0, 

        'close': 100.0

    }])

    

    bars = {c: create_mock_bar(c, current_time, 100.0, 100.0) for c in initial_holdings + [test_code_new]}

    strategy.next(bars)

    

    # 验证 s1 被卖出

    sell_call_found = any(call[0][0].code == 's1' and call[0][0].direction.name == 'SHORT' 

                         for call in mock_broker.submit_order.call_args_list)

    assert sell_call_found

    assert 's1' not in strategy.holdings_order

    

    # 执行买入

    next_day_time = datetime(2026, 2, 5, 9, 30, 0)

    next_bars = {test_code_new: create_mock_bar(test_code_new, next_day_time, 100.0, 100.0)}

    strategy.next(next_bars)

    

    assert strategy.holdings_order == ['s2', 's3', 's4', 's5', test_code_new]


    
    