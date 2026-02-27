import unittest
from datetime import datetime
import numpy as np
from stock.strategy.ksp_core import KSPCore

class TestKSPCore(unittest.TestCase):
    def setUp(self):
        self.mock_selector = type('MockSelector', (), {'select': lambda date: ['SH600000', 'SZ000001']})
        self.strategy = KSPCore(
            selector_obj=self.mock_selector,
            slots=2,
            sell_rank=100,
            take_profit=0.099,
            stop_loss=-0.05
        )
        self.test_date = datetime(2025, 1, 1)

    def test_select_targets(self):
        targets = self.strategy.select_targets(self.test_date, {})
        self.assertEqual(targets, ['SH600000', 'SZ000001'])

    def test_should_exit_stop_loss(self):
        # 成本 100，现价 94 (跌 6%)，触发 5% 止损
        reason = self.strategy.should_exit('SH600000', 100.0, 94.0, self.test_date, {})
        self.assertTrue(reason is not None)
        self.assertIn("stop_loss", reason)

    def test_should_exit_take_profit(self):
        # 成本 100，现价 110 (涨 10%)，触发 9.9% 止盈
        reason = self.strategy.should_exit('SH600000', 100.0, 110.0, self.test_date, {})
        self.assertTrue(reason is not None)
        self.assertIn("take_profit", reason)

    def test_should_exit_rank(self):
        # 排名 150，超过 sell_rank=100
        context = {'rank': 150}
        reason = self.strategy.should_exit('SH600000', 100.0, 105.0, self.test_date, context)
        self.assertTrue(reason is not None)
        self.assertIn("rank_150_exceeds_100", reason)

    def test_should_not_exit(self):
        # 涨 5%，排名 50，均未触发出场
        context = {'rank': 50}
        reason = self.strategy.should_exit('SH600000', 100.0, 105.0, self.test_date, context)
        self.assertIsNone(reason)

    def test_get_execution_price_poc(self):
        # 存在有效 POC
        context = {'poc': 102.5, 'open': 101.0}
        price = self.strategy.get_execution_price('SH600000', self.test_date, context)
        self.assertEqual(price, 102.5)

    def test_get_execution_price_fallback(self):
        # POC 无效，回退至开盘价
        context = {'poc': None, 'open': 101.0}
        price = self.strategy.get_execution_price('SH600000', self.test_date, context)
        self.assertEqual(price, 101.0)

    def test_filter_candidates(self):
        candidates = ['A', 'B', 'C']
        # A 排名 50 (通过), B 排名 150 (过滤), C 无数据 (过滤)
        context = {
            'rank_map': {'A': 50, 'B': 150}
        }
        filtered = self.strategy.filter_candidates(candidates, self.test_date, context)
        self.assertEqual(filtered, ['A'])

if __name__ == '__main__':
    unittest.main()
