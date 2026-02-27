from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from .base import BaseStrategy
from .rules import EntryRule, ExitRule

class ModularKSPCore(BaseStrategy):
    """
    模块化 KSP 策略核心：
    支持注入多个买入准入规则 (EntryRules) 和 卖出退出规则 (ExitRules)
    """

    def __init__(self, 
                 selector_obj: Any,
                 entry_rules: List[EntryRule],
                 exit_rules: List[ExitRule],
                 slots: int = 9):
        self.selector = selector_obj
        self.entry_rules = entry_rules
        self.exit_rules = exit_rules
        self.slots = slots

    def select_targets(self, date: datetime, context: Dict[str, Any]) -> List[str]:
        """
        委派股票池选择器获取初步候选
        """
        if self.selector and hasattr(self.selector, 'select'):
            return self.selector.select(date)
        return []

    def should_exit(self, 
                    code: str, 
                    entry_price: float, 
                    current_price: float, 
                    date: datetime, 
                    context: Dict[str, Any]) -> Optional[str]:
        """
        遍历所有退出规则，任一规则满足即退出
        """
        # 预处理 context 中的排名，防止 0 值干扰
        def clean_rank(r):
            if r is None or np.isnan(r) or r <= 0:
                return 5000.0
            return float(r)
            
        exit_context = context.copy()
        if 'rank_5d' in exit_context: 
            exit_context['ksp_sum_5d_rank'] = clean_rank(exit_context['rank_5d'])
        if 'rank_10d' in exit_context: 
            exit_context['ksp_sum_10d_rank'] = clean_rank(exit_context['rank_10d'])
        if 'rank' in exit_context: 
            exit_context['rank'] = clean_rank(exit_context['rank'])

        for rule in self.exit_rules:
            reason = rule.check(code, entry_price, current_price, date, exit_context)
            if reason:
                return reason
        return None

    def filter_candidates(self, candidates: List[str], date: datetime, context: Dict[str, Any]) -> List[str]:
        """
        遍历所有准入规则，必须全部满足才买入。
        新增逻辑：对满足规则的个股按排名进行二次排序，确保即使有很多候选也只取最好的进入执行阶段。
        """
        passed_with_ranks = []
        rank_map = context.get('rank_map', {})
        rank_5d_map = context.get('rank_5d_map', {})
        rank_10d_map = context.get('rank_10d_map', {})

        def clean_rank(r):
            if r is None or np.isnan(r) or r <= 0:
                return 5000.0
            return float(r)

        for code in candidates:
            stock_context = context.copy()
            stock_context['code'] = code
            
            # 获取排名用于规则检查和后续排序
            r_val = clean_rank(stock_context.get('rank') or rank_map.get(code))
            stock_context['rank'] = r_val
            stock_context['ksp_sum_5d_rank'] = clean_rank(stock_context.get('ksp_sum_5d_rank') or rank_5d_map.get(code) or context.get('rank_5d'))
            stock_context['ksp_sum_10d_rank'] = clean_rank(stock_context.get('ksp_sum_10d_rank') or rank_10d_map.get(code) or context.get('rank_10d'))
            
            all_pass = True
            for rule in self.entry_rules:
                if not rule.check(code, date, stock_context):
                    all_pass = False
                    break
            
            if all_pass:
                passed_with_ranks.append((code, r_val))
        
        # 按排名(从小到大)排序，取最好的 top_n
        # 这里的 top_n 可以复用策略的 slots，或者定义一个新的参数。
        # 为了灵活，我们先全量返回，但在 KSPStrategy 的买入循环中会受到 slots 的限制。
        # 如果要严格实现“满足条件的里面再挑最好的前 N 名”，就在这里排序：
        passed_with_ranks.sort(key=lambda x: x[1])
        
        return [item[0] for i, item in enumerate(passed_with_ranks)]

    def get_execution_price(self, code: str, date: datetime, context: Dict[str, Any]) -> float:
        """
        执行价格决策：直接使用开盘价买入。
        """
        open_price = context.get('open', 0.0)
        return float(open_price)
