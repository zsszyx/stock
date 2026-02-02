from typing import Type, List
import pandas as pd
from .data_feed import DataFeed
from .broker import Broker
from .strategy import Strategy
from .analyzer import Analyzer

class BacktestEngine:
    def __init__(self, initial_cash=100000.0):
        self.broker = Broker(cash=initial_cash)
        self.data_feed: DataFeed = None
        self.strategy_class: Type[Strategy] = None
        self.strategy_instance: Strategy = None
        self.analyzer = Analyzer(self.broker) # Initialize Analyzer
        
    def set_data_feed(self, feed: DataFeed):
        self.data_feed = feed

    def add_strategy(self, strategy_cls: Type[Strategy], **kwargs):
        self.strategy_class = strategy_cls
        self.strategy_kwargs = kwargs

    def run(self, codes: List[str], start_date: str, end_date: str):
        if not self.data_feed or not self.strategy_class:
            raise ValueError("DataFeed and Strategy must be set.")

        # Initialize Strategy
        self.strategy_instance = self.strategy_class(self.broker, **self.strategy_kwargs)
        self.strategy_instance.initialize()

        # Load Data
        print("Loading data...")
        self.data_feed.load_data(codes, start_date, end_date)
        
        print("Starting Backtest...")
        count = 0
        for bars in self.data_feed:
            # 1. Update Strategy History
            self.strategy_instance.update_data(bars)
            
            # 2. Broker processes pending orders (fills them based on current bars)
            self.broker.process_orders(bars)
            
            # 3. Strategy Logic
            self.strategy_instance.next(bars)
            
            # 4. Record Portfolio Value (for analysis)
            current_prices = {c: b.close for c, b in bars.items()}
            self.broker.value_history.append(self.broker.get_portfolio_value(current_prices))
            
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} bars...")
                
        print("Backtest Complete.")
        self._print_results()

    def _print_results(self):
        metrics = self.analyzer.get_performance_metrics()
        
        print("-" * 30)
        print("PERFORMANCE REPORT")
        print("-" * 30)
        for key, value in metrics.items():
            print(f"{key:<25}: {value:.2f}")
        print("-" * 30)
