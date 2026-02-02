from abc import ABC, abstractmethod
from typing import Dict, List
from .models import Bar, Order, OrderType, Direction
from .broker import Broker

class Strategy(ABC):
    def __init__(self, broker: Broker):
        self.broker = broker
        # Mapping code -> list of bars (history)
        self.data_history: Dict[str, List[Bar]] = {} 

    @abstractmethod
    def initialize(self):
        """Called before backtest starts"""
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
