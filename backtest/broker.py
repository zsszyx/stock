from typing import List, Dict, Optional
from .models import Order, Trade, Position, OrderStatus, OrderType, Direction, Bar

class Broker:
    def __init__(self, cash: float = 100000.0, commission_rate: float = 0.0003):
        self.initial_cash = cash
        self.cash = cash
        self.commission_rate = commission_rate
        self.positions: Dict[str, Position] = {}
        self.active_orders: List[Order] = []
        self.trades: List[Trade] = []
        self.value_history: List[float] = [] # For plotting/analysis

    def submit_order(self, order: Order):
        order.status = OrderStatus.SUBMITTED
        self.active_orders.append(order)

    def get_position(self, code: str) -> Position:
        if code not in self.positions:
            self.positions[code] = Position(code=code, quantity=0, avg_price=0.0)
        return self.positions[code]

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        pos_value = 0.0
        for code, pos in self.positions.items():
            price = current_prices.get(code, pos.avg_price) # Fallback to avg if no current price
            pos_value += pos.quantity * price
        return self.cash + pos_value

    def process_orders(self, current_bars: Dict[str, Bar]):
        """
        Match orders against current bars.
        Simulate execution at Close price for simplicity (or Open of next bar in real life).
        Using Close of current bar implies we trade immediately at signal generation time.
        """
        filled_orders = []
        
        for order in self.active_orders:
            if order.code not in current_bars:
                continue
            
            bar = current_bars[order.code]
            execute_price = 0.0
            
            # 1. Determine Execution Price
            if order.type == OrderType.MARKET:
                execute_price = bar.close
            elif order.type == OrderType.LIMIT:
                if order.direction == Direction.LONG and bar.low <= order.price:
                    execute_price = order.price # Simplified
                elif order.direction == Direction.SHORT and bar.high >= order.price:
                    execute_price = order.price
                else:
                    continue # Not triggered
            
            # 2. Check Validity (Money/Position)
            cost = execute_price * order.quantity
            commission = cost * self.commission_rate
            
            if order.direction == Direction.LONG:
                if self.cash >= (cost + commission):
                    self._execute_trade(order, execute_price, commission, bar.time)
                    filled_orders.append(order)
                else:
                    # Margin/Cash error, reject or keep pending? Let's reject for now.
                    order.status = OrderStatus.REJECTED
                    filled_orders.append(order)
                    
            elif order.direction == Direction.SHORT:
                # Sell logic (Close position)
                pos = self.get_position(order.code)
                if pos.quantity >= order.quantity:
                    self._execute_trade(order, execute_price, commission, bar.time)
                    filled_orders.append(order)
                else:
                    order.status = OrderStatus.REJECTED
                    filled_orders.append(order)

        # Remove filled/rejected from active
        for order in filled_orders:
            if order in self.active_orders:
                self.active_orders.remove(order)

    def _execute_trade(self, order: Order, price: float, commission: float, time):
        order.status = OrderStatus.FILLED
        order.filled_price = price
        order.filled_time = time
        
        # Update Cash
        if order.direction == Direction.LONG:
            self.cash -= (price * order.quantity + commission)
        else:
            self.cash += (price * order.quantity - commission)
            
        # Update Position
        pos = self.get_position(order.code)
        if order.direction == Direction.LONG:
            # Update Avg Price
            total_cost = (pos.quantity * pos.avg_price) + (order.quantity * price)
            pos.quantity += order.quantity
            pos.avg_price = total_cost / pos.quantity if pos.quantity > 0 else 0
        else:
            pos.quantity -= order.quantity
            # Selling doesn't change avg_price of remaining shares (FIFO/Weighted assumption)
            if pos.quantity == 0:
                pos.avg_price = 0

        # Record Trade
        trade = Trade(
            order_id=order.id,
            code=order.code,
            direction=order.direction,
            quantity=order.quantity,
            price=price,
            time=time,
            commission=commission
        )
        self.trades.append(trade)
