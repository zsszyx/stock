# 股票策略回测框架 (Stock Strategy Backtesting Framework)

这是一个模块化、事件驱动的股票交易策略回测引擎。基于 Python 开发，严格遵循 SOLID 原则，旨在提供高可维护性和强扩展性。

## 📂 项目结构

```text
stock/backtest/
├── analyzer.py    # 绩效分析 (夏普比率, 最大回撤, 胜率等)
├── broker.py      # 订单执行, 持仓管理及佣金逻辑
├── data_feed.py   # 数据加载抽象层 (支持 SQLite 等)
├── engine.py      # 核心事件循环与协调器
├── models.py      # 基础数据结构 (Bar, Order, Trade, Position)
└── strategy.py    # 用户策略基类
```

## 🚀 核心特性

*   **事件驱动架构 (Event-Driven)**: 逐根 K 线模拟真实市场行为，支持复杂的交易逻辑。
*   **数据库集成**: 无缝对接项目现有的 SQLite 数据库 (`stock.db`)。
*   **灵活的策略 API**: 采用类似于 `Backtrader` 或 `Zipline` 的 `initialize` (初始化) 和 `next` (逐K线逻辑) 接口，上手简单。
*   **全面的分析指标**:
    *   **收益**: 总收益率 (Total Return), 年化收益率 (Annualized Return)。
    *   **风险**: 最大回撤 (Max Drawdown), 夏普比率 (Sharpe Ratio), 波动率 (Volatility)。
    *   **交易统计**: 胜率 (Win Rate), 盈亏比 (Profit Factor), 平均盈亏比 (Avg Win/Loss Ratio)。
*   **真实的交易模拟**:
    *   支持市价单 (Market) 和限价单 (Limit)。
    *   自动扣除佣金 (Commission)。
    *   持仓追踪 (采用 FIFO 先进先出原则计算 PnL)。

## 🛠 使用指南

### 1. 定义策略

创建一个继承自 `Strategy` 的类。实现 `initialize` (设置参数) 和 `next` (编写每个时间步的逻辑)。

```python
from stock.backtest.strategy import Strategy

class MySMAStrategy(Strategy):
    def initialize(self):
        self.sma_period = 20

    def next(self, bars):
        for code, bar in bars.items():
            # 获取历史数据
            history = self.data_history[code]
            if len(history) < self.sma_period:
                return

            # 计算指标 (例如: 简单移动平均线 SMA)
            closes = [b.close for b in history[-self.sma_period:]]
            sma = sum(closes) / len(closes)
            
            # 交易逻辑
            pos = self.broker.get_position(code)
            
            # 买入信号
            if bar.close > sma and pos.quantity == 0:
                self.buy(code, 100) # 市价单买入 100 股
                
            # 卖出信号
            elif bar.close < sma and pos.quantity > 0:
                self.sell(code, pos.quantity) # 卖出所有持仓
```

### 2. 配置并运行回测

使用 `BacktestEngine` 将所有组件组装起来。

```python
from stock.backtest.engine import BacktestEngine
from stock.backtest.data_feed import SqliteDataFeed
from stock.sql_op.op import SqlOp

# 1. 设置数据源
sql_op = SqlOp() # 使用项目中现有的 SQL 助手
data_feed = SqliteDataFeed(sql_op, table_name="mintues5")

# 2. 初始化引擎
engine = BacktestEngine(initial_cash=100000.0) # 初始资金 10万
engine.set_data_feed(data_feed)

# 3. 添加策略
engine.add_strategy(MySMAStrategy)

# 4. 运行回测
engine.run(
    codes=['sh.600000'], 
    start_date='2026-01-01', 
    end_date='2026-01-29'
)
```

### 3. 解读结果

回测完成后，引擎会自动在控制台打印详细的绩效报告：

```text
------------------------------
PERFORMANCE REPORT (绩效报告)
------------------------------
Initial Capital          : 100000.00   (初始资金)
Final Equity             : 99875.50    (最终权益)
Total Return (%)         : -0.12       (总收益率)
Annualized Return (%)    : -4.92       (年化收益率)
Max Drawdown (%)         : -0.17       (最大回撤)
Sharpe Ratio             : -12.90      (夏普比率)
Total Trades             : 45.00       (总交易次数)
Total Round Trips        : 22.00       (完整回合交易次数 - 一买一卖)
Win Rate (%)             : 4.55        (胜率)
Profit Factor            : 0.02        (总获利因子 - 总盈利/总亏损)
Avg Win                  : 1.35        (平均单笔盈利)
Avg Loss                 : -4.01       (平均单笔亏损)
Avg Win/Loss Ratio       : 0.34        (平均盈亏比)
------------------------------
```

## 🧩 关键组件详解

### `DataFeed` (数据源)
*   **`SqliteDataFeed`**: 从 `mintues5` 表加载 5 分钟 K 线数据。为了兼顾速度与内存，它会预加载选定日期范围的数据到 Pandas DataFrame，但以生成器 (Generator) 的方式逐个时间步 (`yield`) 返回数据给引擎。

### `Broker` (经纪人)
*   **订单撮合 (Order Matching)**: 当前实现较为简单，默认在信号触发的**同一根 K 线**的 `Close` 价格成交 (乐观假设)。
*   **佣金 (Commissions)**: 默认费率为 0.03% (0.0003)。可通过 `Broker(commission_rate=...)` 进行调整。

### `Analyzer` (分析器)
*   **PnL 计算**: 使用 **FIFO (先进先出)** 算法来匹配买单和卖单。这对于准确计算“完整回合交易 (Round Trip)”的盈亏、胜率和获利因子至关重要。

## ⚠️ 注意事项 & 最佳实践

1.  **未来函数 (Lookahead Bias)**: 当前的简单实现是在信号产生的 K 线收盘价成交。为了更严谨的模拟，建议修改逻辑，在**下一根 K 线**的 `Open` 价成交。
2.  **数据质量**: 确保数据库中的数据没有缺失或 NaN 值（框架假定字段都是有效的数值）。
3.  **性能优化**: 引擎是基于 Python 对象循环的。如果进行超大规模回测（数百万根 K 线），可以考虑将核心指标计算向量化，尽管目前的事件驱动模式提供了最大的逻辑灵活性。
