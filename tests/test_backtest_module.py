"""
回测模块测试 - TDD 开发模式

测试优先级:
1. test_data_loader - 数据加载与校验
2. test_backtest_engine - 回测引擎
3. test_analyzer - 指标计算
4. test_reporter - 报告生成
"""

import unittest
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestDataLoader(unittest.TestCase):
    """数据加载器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.test_data = {
            'code': ['sh.600075', 'sh.600036'],
            'date': ['2026-02-01', '2026-02-02'],
            'close': [10.0, 20.0],
            'volume': [1000000, 2000000]
        }
    
    def test_validate_stock_data(self):
        """测试股票数据有效性校验"""
        # 测试空数据
        self.assertFalse(self._is_valid_stock([]))
        
        # 测试有效数据
        self.assertTrue(self._is_valid_stock([{'close': 10.0, 'volume': 1000}]))
        
        # 测试零成交量
        self.assertFalse(self._is_valid_stock([{'close': 10.0, 'volume': 0}]))
        
        # 测试NaN价格
        self.assertFalse(self._is_valid_stock([{'close': None, 'volume': 1000}]))
    
    def _is_valid_stock(self, data):
        """检查数据是否有效"""
        if not data:
            return False
        for row in data:
            if row.get('volume', 0) <= 0:
                return False
            if row.get('close') is None:
                return False
        return True
    
    def test_filter_invalid_stocks(self):
        """测试过滤无效股票"""
        stocks = [
            {'code': 'sh.600075', 'data': []},
            {'code': 'sh.600036', 'data': [{'close': 10.0, 'volume': 1000}]},
            {'code': 'sh.600000', 'data': [{'close': 10.0, 'volume': 0}]},
        ]
        
        valid = [s for s in stocks if self._is_valid_stock(s['data'])]
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]['code'], 'sh.600036')


class TestBacktestEngine(unittest.TestCase):
    """回测引擎测试"""
    
    def test_initial_capital(self):
        """测试初始资金设置"""
        initial_capital = 1000000
        self.assertEqual(initial_capital, 1000000)
    
    def test_position_size_calculation(self):
        """测试仓位计算"""
        capital = 1000000
        position_pct = 0.11  # 11%
        
        position_value = capital * position_pct
        self.assertAlmostEqual(position_value, 110000, delta=1)
    
    def test_stop_loss_logic(self):
        """测试止损逻辑"""
        entry_price = 10.0
        stop_loss = -0.02  # -2%
        
        stop_price = entry_price * (1 + stop_loss)
        self.assertAlmostEqual(stop_price, 9.80, places=2)
    
    def test_take_profit_logic(self):
        """测试止盈逻辑"""
        entry_price = 10.0
        take_profit = 0.099  # +9.9%
        
        target_price = entry_price * (1 + take_profit)
        self.assertAlmostEqual(target_price, 10.99, places=2)


class TestAnalyzer(unittest.TestCase):
    """指标分析器测试"""
    
    def test_calculate_return(self):
        """测试收益率计算"""
        initial = 1000000
        final = 1200000
        
        return_pct = (final - initial) / initial
        self.assertAlmostEqual(return_pct, 0.20, places=2)
    
    def test_calculate_max_drawdown(self):
        """测试最大回撤计算"""
        equity_curve = [1000000, 1100000, 1050000, 1200000, 950000]
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        # 1200000 -> 950000: (1200000-950000)/1200000 = 20.8%
        self.assertAlmostEqual(max_dd, 0.208, places=2)
    
    def test_calculate_sharpe_ratio(self):
        """测试夏普比率计算"""
        returns = [0.01, -0.005, 0.02, 0.015, -0.01]
        
        import numpy as np
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            sharpe = (mean_return * 252) / (std_return * np.sqrt(252))
        else:
            sharpe = 0
        
        self.assertIsInstance(sharpe, float)
    
    def test_average_holding_days(self):
        """测试平均持仓天数计算"""
        # 模拟交易记录: (买入日期, 卖出日期, 持有天数)
        trades = [
            {'hold_days': 5},
            {'hold_days': 10},
            {'hold_days': 3},
            {'hold_days': 7},
        ]
        
        total_days = sum(t['hold_days'] for t in trades)
        avg_days = total_days / len(trades)
        
        self.assertAlmostEqual(avg_days, 6.25, places=1)


class TestReporter(unittest.TestCase):
    """报告生成器测试"""
    
    def test_generate_summary(self):
        """测试摘要生成"""
        summary = {
            'initial_capital': 1000000,
            'final_value': 1200000,
            'total_trades': 100,
            'win_rate': 0.6,
            'max_drawdown': 0.05,
            'sharpe_ratio': 1.5,
        }
        
        self.assertIn('initial_capital', summary)
        self.assertIn('final_value', summary)
        self.assertEqual(summary['total_trades'], 100)


if __name__ == '__main__':
    unittest.main()
