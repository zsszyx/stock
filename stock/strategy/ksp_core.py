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
                 entry_rank: int = 100,
                 sell_rank: int = 400,
                 take_profit: float = 0.099,
                 stop_loss: float = -0.02):
        self.selector = selector_obj
        self.slots = slots
        self.entry_rank = entry_rank
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
        KSP 退出逻辑判断：5日排名不再优于10日排名时退出
        """
        if entry_price <= 0:
            return None
        
        # 1. 获取排名数据
        r5 = context.get('rank_5d')
        r10 = context.get('rank_10d')
        
        # 如果缺少排名数据，保守起见不在此处退出 (或根据需要调整)
        if r5 is None or r10 is None or np.isnan(r5) or np.isnan(r10):
            return None
            
        # 2. 动能反转逻辑：5日排名不再优于10日排名 (即 r5 >= r10)
        # 这里的排名是数值，数值越小排名越高
        if r5 >= r10:
            return f"momentum_flip_r5_{r5:.0f}_r10_{r10:.0f}"
        
        # 3. 硬门槛：如果 5日排名跌出 sell_rank (可选，保持一定的绝对质量控制)
        if r5 > self.sell_rank:
            return f"rank_5d_{r5:.0f}_exceeds_{self.sell_rank}"
        
        return None

    def get_execution_price(self, code: str, date: datetime, context: Dict[str, Any]) -> float:
        """
        执行价格决策：
        1. 严格使用 POC 价格作为限价单价格，以避免过高的买入点
        2. 仅当 POC 无效时回退至开盘价
        """
        poc_price = context.get('poc')
        open_price = context.get('open', 0.0)
        
        if poc_price is not None and not np.isnan(poc_price) and poc_price > 0.01:
            return float(poc_price)
        return float(open_price)

    def filter_candidates(self, candidates: List[str], date: datetime, context: Dict[str, Any]) -> List[str]:
        """
        按 KSP 排名规则过滤掉排名已超过 entry_rank 的标的 (买入时的硬门槛)
        :param context: Dict[code, rank]
        """
        filtered = []
        for code in candidates:
            # 在 Backtrader 中，context 会传入所有数据流的最新排名
            rank_map = context.get('rank_map', {})
            rank = rank_map.get(code)
            
            if rank is not None and not np.isnan(rank):
                if rank <= self.entry_rank:
                    filtered.append(code)
            else:
                # 如果没有排名数据，保持现状或根据策略决定（此处保守起见跳过）
                pass
        return filtered
