from typing import Type, List, Optional
from .broker import Broker
from .strategy import Strategy
from .analyzer import Analyzer
from .data_feed import DataFeed
from stock.data.repository import DataRepository
from stock.data.context import MarketContext, Minutes5Context

class BacktestEngine:
    def __init__(self, initial_cash=100000.0):
        self.broker = Broker(cash=initial_cash)
        self.data_repository = DataRepository()
        self.data_feed: DataFeed = None
        self.strategy_class: Type[Strategy] = None
        self.strategy_kwargs: dict = {}
        self.analyzer = Analyzer(self.broker)
        
    def set_data_feed(self, feed: DataFeed):
        self.data_feed = feed

    def add_strategy(self, strategy_cls: Type[Strategy], **kwargs):
        self.strategy_class = strategy_cls
        self.strategy_kwargs = kwargs

    def run(self, codes: Optional[List[str]], start_date: str, end_date: str):
        # 1. 加载数据并构建上下文
        full_df = self.data_repository.load_minutes5(start_date, end_date)
        m5_ctx = Minutes5Context(full_df)
        context = MarketContext(minutes5=m5_ctx)
        
        # 2. 实例化策略
        strategy = self.strategy_class(self.broker, context, **self.strategy_kwargs)
        strategy.initialize()

        # 3. 确定 Universe
        universe = codes if codes else strategy.screen(start_date, end_date)
        
        # 4. 配置 DataFeed 并运行
        self.data_feed.set_universe(universe, start_date, end_date)
        
        print(f"Engine: Starting Backtest [{start_date} -> {end_date}]")
        for bars in self.data_feed:
            # 更新上下文中的当前时间，供选股器引用
            context.current_date = list(bars.values())[0].time
            
            strategy.update_data(bars)
            self.broker.process_orders(bars)
            strategy.next(bars)
            
            current_prices = {c: b.close for c, b in bars.items()}
            self.broker.value_history.append(self.broker.get_portfolio_value(current_prices))
                
        print("Engine: Backtest Complete.")
        self._print_results()

    def _print_results(self):
        metrics = self.analyzer.get_performance_metrics()
        print("-" * 30 + "\nPERFORMANCE REPORT\n" + "-" * 30)
        for k, v in metrics.items(): print(f"{k:<25}: {v:.2f}")
        print("-" * 30)