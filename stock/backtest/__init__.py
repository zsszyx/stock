"""
Stock Backtest Package (Backtrader Refactored)

核心组件:
- BTBacktester: 基于 Backtrader 的回测执行引擎
- KSPPandasData: 支持 KSP 因子的数据加载类
- KSPStrategy: 支持 KSPCore 策略注入的适配器
"""

from .bt_backtester import BTBacktester, KSPPandasData
from .ksp_strategy import KSPStrategy

__all__ = [
    'BTBacktester',
    'KSPPandasData',
    'KSPStrategy',
]
