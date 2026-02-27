from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

class BaseRule(ABC):
    """
    规则基类
    """
    def __init__(self, name: str):
        self.name = name

class EntryRule(BaseRule):
    """
    买入准入规则接口
    """
    @abstractmethod
    def check(self, code: str, date: datetime, context: Dict[str, Any]) -> bool:
        pass

class ExitRule(BaseRule):
    """
    卖出退出规则接口
    """
    @abstractmethod
    def check(self, code: str, entry_price: float, current_price: float, 
              date: datetime, context: Dict[str, Any]) -> Optional[str]:
        """
        返回退出原因字符串，如果不退出则返回 None
        """
        pass

# --- 具体买入规则实现 ---

class RankEntryRule(EntryRule):
    """
    基于绝对排名的准入规则
    """
    def __init__(self, rank_col: str, threshold: int):
        super().__init__(f"Rank_{rank_col}_Under_{threshold}")
        self.rank_col = rank_col
        self.threshold = threshold

    def check(self, code: str, date: datetime, context: Dict[str, Any]) -> bool:
        # 1. 尝试直接从 context 拿对应的列名
        rank = context.get(self.rank_col)
        
        # 2. 如果没有，看是否 context['code'] 匹配，如果是则拿 context['rank']
        if rank is None and context.get('code') == code:
            rank = context.get('rank')
            
        # 3. 最后尝试从 rank_map 拿
        if rank is None:
            rank = context.get('rank_map', {}).get(code)
            
        if rank is not None and not np.isnan(rank):
            return rank <= self.threshold
        return False

class RangeRankEntryRule(EntryRule):
    """
    基于排名区间的准入规则 (次优逻辑：避开 D1，选择 D2-D3)
    """
    def __init__(self, rank_col: str, min_rank: int, max_rank: int):
        super().__init__(f"Rank_{rank_col}_Between_{min_rank}_{max_rank}")
        self.rank_col = rank_col
        self.min_rank = min_rank
        self.max_rank = max_rank

    def check(self, code: str, date: datetime, context: Dict[str, Any]) -> bool:
        rank = context.get(self.rank_col) or context.get('rank')
        if rank is not None and not np.isnan(rank):
            return self.min_rank <= rank <= self.max_rank
        return False

class VolatilityConvergenceRule(EntryRule):
    """
    基于波动收敛的准入规则 (振幅收缩)
    """
    def __init__(self, threshold: float = 0.025):
        super().__init__(f"Volatility_Convergence_Under_{threshold:.1%}")
        self.threshold = threshold

    def check(self, code: str, date: datetime, context: Dict[str, Any]) -> bool:
        # 从 context 获取振幅，如果不存在则计算
        amplitude = context.get('amplitude')
        
        if amplitude is None:
            high = context.get('high')
            low = context.get('low')
            close = context.get('close')
            if high and low and close and close > 0:
                amplitude = (high - low) / close
        
        if amplitude is not None and not np.isnan(amplitude):
            return amplitude <= self.threshold
        return False

class VolumeRatioEntryRule(EntryRule):
    """
    基于量比异动的准入规则 (U型放量启动)
    """
    def __init__(self, threshold: float = 1.5, window: int = 5):
        super().__init__(f"Volume_Ratio_Over_{threshold}")
        self.threshold = threshold
        self.window = window

    def check(self, code: str, date: datetime, context: Dict[str, Any]) -> bool:
        vol_ratio = context.get('vol_ratio')
        if vol_ratio is not None and not np.isnan(vol_ratio):
            return vol_ratio >= self.threshold
        return False

class MovingAverageBiasRule(EntryRule):
    """
    基于均线乖离率的准入规则 (控制回踩/不追高)
    Bias = (Price - MA) / MA
    """
    def __init__(self, window: int = 20, max_bias: float = 0.05, min_bias: float = -0.03):
        super().__init__(f"MA{window}_Bias_{min_bias:.1%}_to_{max_bias:.1%}")
        self.window = window
        self.max_bias = max_bias
        self.min_bias = min_bias

    def check(self, code: str, date: datetime, context: Dict[str, Any]) -> bool:
        bias = context.get(f'bias_{self.window}')
        
        # 如果 context 中没有预计算好的 bias，规则尝试自行计算 (需要 context 中有 ma 数据)
        if bias is None:
            ma = context.get(f'ma_{self.window}')
            close = context.get('close')
            if ma and close and ma > 0:
                bias = (close - ma) / ma
        
        if bias is not None and not np.isnan(bias):
            return self.min_bias <= bias <= self.max_bias
        return False

class MomentumEntryRule(EntryRule):
    """
    基于动能趋势的准入规则 (r5 < r10)
    """
    def __init__(self, short_rank: str = 'ksp_sum_5d_rank', long_rank: str = 'ksp_sum_10d_rank'):
        super().__init__("KSP_Momentum_Trend")
        self.short_rank = short_rank
        self.long_rank = long_rank

    def check(self, code: str, date: datetime, context: Dict[str, Any]) -> bool:
        r_short = context.get(self.short_rank)
        r_long = context.get(self.long_rank)
        
        # if date.strftime('%Y-%m-%d') == '2025-01-02':
        #    print(f"DEBUG: Rule {self.name} for {code}: short({self.short_rank})={r_short}, long({self.long_rank})={r_long}")
            
        if r_short is not None and r_long is not None and not np.isnan(r_short) and not np.isnan(r_long):
            return r_short < r_long
        return False

# --- 具体卖出规则实现 ---

class StopLossRule(ExitRule):
    """
    固定止损规则
    """
    def __init__(self, threshold: float = -0.02):
        super().__init__("Stop_Loss")
        self.threshold = threshold

    def check(self, code: str, entry_price: float, current_price: float, 
              date: datetime, context: Dict[str, Any]) -> Optional[str]:
        if entry_price <= 0: return None
        pnl_pct = (current_price - entry_price) / entry_price
        if pnl_pct <= self.threshold:
            return f"stop_loss_{pnl_pct:.2%}"
        return None

class TakeProfitRule(ExitRule):
    """
    固定止盈规则
    """
    def __init__(self, threshold: float = 0.099):
        super().__init__("Take_Profit")
        self.threshold = threshold

    def check(self, code: str, entry_price: float, current_price: float, 
              date: datetime, context: Dict[str, Any]) -> Optional[str]:
        if entry_price <= 0: return None
        pnl_pct = (current_price - entry_price) / entry_price
        if pnl_pct >= self.threshold:
            return f"take_profit_{pnl_pct:.2%}"
        return None

class RankExitRule(ExitRule):
    """
    绝对排名退出规则
    """
    def __init__(self, rank_col: str, threshold: int):
        super().__init__(f"Rank_{rank_col}_Exceeds_{threshold}")
        self.rank_col = rank_col
        self.threshold = threshold

    def check(self, code: str, entry_price: float, current_price: float, 
              date: datetime, context: Dict[str, Any]) -> Optional[str]:
        rank = context.get('rank') or context.get(self.rank_col)
        if rank is not None and not np.isnan(rank):
            if rank > self.threshold:
                return f"rank_{rank:.0f}_exceeds_{self.threshold}"
        return None

class BottomRankExitRule(ExitRule):
    """
    后 20% (D9-D10) 排名退出规则
    """
    def __init__(self, rank_col: str = 'ksp_sum_5d_rank', bottom_threshold: int = 3500):
        super().__init__(f"Bottom_{rank_col}_{bottom_threshold}")
        self.rank_col = rank_col
        self.bottom_threshold = bottom_threshold

    def check(self, code: str, entry_price: float, current_price: float, 
              date: datetime, context: Dict[str, Any]) -> Optional[str]:
        # 优先使用具体的列，其次尝试通用键
        rank = context.get(self.rank_col) or context.get('rank_5d') or context.get('rank')
        
        if rank is not None and not np.isnan(rank):
            if rank >= self.bottom_threshold:
                return f"bottom_rank_{rank:.0f}_in_tail"
        return None

class MomentumFlipRule(ExitRule):
    """
    动能反转退出规则 (r5 >= r10)
    """
    def __init__(self, short_rank: str = 'ksp_sum_5d_rank', long_rank: str = 'ksp_sum_10d_rank'):
        super().__init__("Momentum_Flip")
        self.short_rank = short_rank
        self.long_rank = long_rank

    def check(self, code: str, entry_price: float, current_price: float, 
              date: datetime, context: Dict[str, Any]) -> Optional[str]:
        # 尝试多种可能的 key
        r_short = context.get(self.short_rank) or context.get('rank_5d') or context.get('ksp_sum_5d_rank')
        r_long = context.get(self.long_rank) or context.get('rank_10d') or context.get('ksp_sum_10d_rank')
        
        if r_short is not None and r_long is not None and not np.isnan(r_short) and not np.isnan(r_long):
            if r_short >= r_long:
                return f"momentum_flip_r5_{r_short:.0f}_r10_{r_long:.0f}"
        return None
