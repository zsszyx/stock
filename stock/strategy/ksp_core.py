from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from .base import BaseStrategy

class KSPCore(BaseStrategy):
    """
    KSP 策略核心逻辑实现：
    1. 模块化止盈止损与排名退出
    2. 基于 POC 的买入执行价决策
    3. 支持注入任意股票池选择器
    """

    def __init__(self, 
                 selector_obj: Any,
                 slots: int = 9,
                 sell_rank: int = 400,
                 take_profit: float = 0.10,
                 stop_loss: float = -0.02):
        self.selector = selector_obj
        self.slots = slots
        self.sell_rank = sell_rank
        self.take_profit = take_profit
        self.stop_loss = stop_loss

    def select_targets(self, date: datetime, context: Dict[str, Any]) -> List[str]:
        """
        委派股票池选择器（如 ConceptSelector）获取初步候选
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
        KSP 退出逻辑判断：止损、止盈或排名超限
        :param context: 必须包含 'rank' 键（当前股票的排名）
        """
        if entry_price <= 0:
            return None
        
        pnl_pct = (current_price - entry_price) / entry_price
        
        # 1. 强制止损
        if pnl_pct <= self.stop_loss:
            return f"stop_loss_{pnl_pct:.2%}"
        
        # 2. 止盈
        if pnl_pct >= self.take_profit:
            return f"take_profit_{pnl_pct:.2%}"
        
        # 3. 排名卖出逻辑 (KSP核心机制)
        rank = context.get('rank')
        if rank is not None and not np.isnan(rank):
            if rank > self.sell_rank:
                return f"rank_{rank:.0f}_exceeds_{self.sell_rank}"
        
        return None

    def get_execution_price(self, code: str, date: datetime, context: Dict[str, Any]) -> float:
        """
        执行价格优先使用 POC，无 POC 时回退至开盘价
        :param context: 包含 'poc' 和 'open'
        """
        poc_price = context.get('poc')
        open_price = context.get('open', 0.0)
        
        if poc_price is not None and not np.isnan(poc_price) and poc_price > 0.01:
            return float(poc_price)
        return float(open_price)

    def filter_candidates(self, candidates: List[str], date: datetime, context: Dict[str, Any]) -> List[str]:
        """
        按 KSP 排名规则过滤掉排名已超过 sell_rank 的标的
        :param context: Dict[code, rank]
        """
        filtered = []
        for code in candidates:
            # 在 Backtrader 中，context 会传入所有数据流的最新排名
            rank_map = context.get('rank_map', {})
            rank = rank_map.get(code)
            
            if rank is not None and not np.isnan(rank):
                if rank <= self.sell_rank:
                    filtered.append(code)
            else:
                # 如果没有排名数据，保持现状或根据策略决定（此处保守起见跳过）
                pass
        return filtered
