# Stock Strategy Backtesting Framework

A modular, event-driven backtesting engine designed for stock trading strategies. Built with Python, adhering to SOLID principles for high maintainability and extensibility.

## 📂 Project Structure

```text
stock/backtest/
├── analyzer.py    # Performance analysis (Sharpe, Drawdown, Win Rate, etc.)
├── broker.py      # Order execution, position management, and commission logic
├── data_feed.py   # Data loading abstraction (supports SQLite, etc.)
├── engine.py      # Core event loop and coordination
├── models.py      # Data structures (Bar, Order, Trade, Position)
└── strategy.py    # Base class for user-defined strategies
```

## 🚀 Features

*   **Event-Driven Architecture**: Simulates realistic market behavior bar-by-bar.
*   **Database Integration**: Seamlessly loads data from the project's SQLite database (`stock.db`).
*   **Flexible Strategy API**: Simple `initialize` and `next` methods familiar to users of `Backtrader` or `Zipline`.
*   **Comprehensive Analytics**:
    *   **Returns**: Total Return, Annualized Return.
    *   **Risk**: Max Drawdown, Sharpe Ratio, Volatility.
    *   **Trade Stats**: Win Rate, Profit Factor, Average Win/Loss Ratio.
*   **Realistic Execution**:
    *   Market and Limit Orders.
    *   Commission handling.
    *   Position tracking (FIFO logic for PnL).

## 🛠 Usage Guide

### 1. Define a Strategy

Create a class inheriting from `Strategy`. Implement `initialize` (setup) and `next` (logic per bar).

```python
from stock.backtest.strategy import Strategy

class MySMAStrategy(Strategy):
    def initialize(self):
        self.sma_period = 20

    def next(self, bars):
        for code, bar in bars.items():
            # Access historical data
            history = self.data_history[code]
            if len(history) < self.sma_period:
                return

            # Calculate Indicator (e.g., Simple Moving Average)
            closes = [b.close for b in history[-self.sma_period:]]
            sma = sum(closes) / len(closes)
            
            # Trading Logic
            pos = self.broker.get_position(code)
            
            # Buy Signal
            if bar.close > sma and pos.quantity == 0:
                self.buy(code, 100) # Market Order
                
            # Sell Signal
            elif bar.close < sma and pos.quantity > 0:
                self.sell(code, pos.quantity)
```

### 2. Configure and Run Backtest

Use the `BacktestEngine` to wire everything together.

```python
from stock.backtest.engine import BacktestEngine
from stock.backtest.data_feed import SqliteDataFeed
from stock.sql_op.op import SqlOp

# 1. Setup Data Feed
sql_op = SqlOp() # Your existing SQL helper
data_feed = SqliteDataFeed(sql_op, table_name="mintues5")

# 2. Initialize Engine
engine = BacktestEngine(initial_cash=100000.0)
engine.set_data_feed(data_feed)

# 3. Add Strategy
engine.add_strategy(MySMAStrategy)

# 4. Run
engine.run(
    codes=['sh.600000'], 
    start_date='2026-01-01', 
    end_date='2026-01-29'
)
```

### 3. Interpret Results

The engine automatically prints a performance report upon completion:

```text
------------------------------
PERFORMANCE REPORT
------------------------------
Initial Capital          : 100000.00
Final Equity             : 99875.50
Total Return (%)         : -0.12
Annualized Return (%)    : -4.92
Max Drawdown (%)         : -0.17
Sharpe Ratio             : -12.90
Total Trades             : 45.00
Total Round Trips        : 22.00
Win Rate (%)             : 4.55
Profit Factor            : 0.02
Avg Win                  : 1.35
Avg Loss                 : -4.01
Avg Win/Loss Ratio       : 0.34
------------------------------
```

## 🧩 Key Components

### `DataFeed`
*   **`SqliteDataFeed`**: Loads 5-minute K-line data from `mintues5` table. Optimizes memory by yielding bars generator-style, though it pre-loads the selected date range into a Pandas DataFrame for speed.

### `Broker`
*   **Order Matching**: Currently simulates execution at the `Close` price of the *same* bar where the signal is generated (optimistic). 
*   **Commissions**: Default is 0.03% (0.0003). Can be adjusted in `Broker(commission_rate=...)`.

### `Analyzer`
*   **PnL Calculation**: Uses **FIFO (First-In-First-Out)** matching to pair Buy and Sell orders for accurate "Round Trip" trade analysis. This is crucial for calculating Win Rate and Profit Factor correctly.

## ⚠️ Notes & Best Practices

1.  **Lookahead Bias**: The current simple implementation executes Market orders at the `Close` of the signal bar. For stricter realism, execute at the `Open` of the *next* bar.
2.  **Data Quality**: Ensure your database has no gaps or NaNs (the framework assumes standard numeric fields).
3.  **Performance**: The engine loops through Python objects. For massive-scale optimizations (millions of bars), consider vectorization, though this event-driven approach offers maximum logic flexibility.
