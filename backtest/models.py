from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class OrderStatus(Enum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class Bar:
    """Standard OHLCV Bar"""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    code: str = ""

@dataclass
class Order:
    """Order to be executed by the Broker"""
    code: str
    direction: Direction
    type: OrderType
    quantity: float
    price: Optional[float] = None  # None for Market Order
    status: OrderStatus = OrderStatus.CREATED
    created_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None
    filled_price: float = 0.0
    id: str = field(default_factory=lambda: str(datetime.now().timestamp()))

@dataclass
class Trade:
    """Record of a completed transaction"""
    order_id: str
    code: str
    direction: Direction
    quantity: float
    price: float
    time: datetime
    commission: float = 0.0

@dataclass
class Position:
    """Current holding of a stock"""
    code: str
    quantity: float = 0.0
    avg_price: float = 0.0
    
    @property
    def market_value(self):
        # Requires current price update
        return 0.0
