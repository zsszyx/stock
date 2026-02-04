from abc import ABC, abstractmethod
from typing import Dict, List
from datetime import datetime
import pandas as pd
from .models import Bar, Order, OrderType, Direction
from .broker import Broker
from stock.data.context import MarketContext

class Strategy(ABC):
    def __init__(self, broker: Broker, context: MarketContext):
        self.broker = broker
        self.context = context
        # Mapping code -> list of bars (history)
        self.data_history: Dict[str, List[Bar]] = {} 

    @abstractmethod
    def initialize(self):
        """Called before backtest starts"""
        pass

    @abstractmethod
    def screen(self, start_date: str, end_date: str) -> List[str]:
        """
        选股逻辑接口：接受日期范围，返回该期间符合条件的股票池列表。
        用于回测开始前确定 Universe。
        """
        pass

    @abstractmethod
    def on_screen(self, date: datetime) -> pd.DataFrame:
        """
        每日选股接口：返回当日选中的股票及因子。
        """
        pass

    @abstractmethod
    def next(self, bars: Dict[str, Bar]):
        """Called on every new time step"""
        pass
    
    def update_data(self, bars: Dict[str, Bar]):
        """Internal method to update history"""
        for code, bar in bars.items():
            if code not in self.data_history:
                self.data_history[code] = []
            self.data_history[code].append(bar)

    def buy(self, code: str, quantity: float, price: float = None):
        """
        Place a Buy order.
        If price is None, Market Order. Else Limit Order.
        """
        order_type = OrderType.LIMIT if price else OrderType.MARKET
        order = Order(
            code=code,
            direction=Direction.LONG,
            type=order_type,
            quantity=quantity,
            price=price
        )
        self.broker.submit_order(order)
        return order

    def sell(self, code: str, quantity: float, price: float = None):
        """
        Place a Sell order.
        """
        order_type = OrderType.LIMIT if price else OrderType.MARKET
        order = Order(
            code=code,
            direction=Direction.SHORT,
            type=order_type,
            quantity=quantity,
            price=price
        )
        self.broker.submit_order(order)
        return order

    @property
    def position(self):
        return self.broker.positions
