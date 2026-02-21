from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

class BaseStrategy(ABC):
    """
    策略抽象基类：定义所有交易策略必须实现的标准接口
    """

    @abstractmethod
    def select_targets(self, date: datetime, context: Dict[str, Any]) -> List[str]:
        """
        根据日期和上下文选择目标股票池
        :param date: 当前交易日期
        :param context: 上下文数据（如行情、排名、概念板块等）
        :return: 目标股票代码列表
        """
        pass

    @abstractmethod
    def should_exit(self, code: str, entry_price: float, current_price: float, 
                    date: datetime, context: Dict[str, Any]) -> Optional[str]:
        """
        判断是否应当平仓（退出当前持仓）
        :param code: 股票代码
        :param entry_price: 持仓成本价
        :param current_price: 当前行情价格
        :param date: 当前交易日期
        :param context: 上下文数据（如排名等）
        :return: 如果退出，返回退出原因；否则返回 None
        """
        pass

    @abstractmethod
    def get_execution_price(self, code: str, date: datetime, context: Dict[str, Any]) -> float:
        """
        根据策略逻辑获取理想的执行价格（如 POC、开盘价等）
        :param code: 股票代码
        :param date: 当前交易日期
        :param context: 上下文数据
        :return: 预期的买入/卖出价格
        """
        pass

    def filter_candidates(self, candidates: List[str], date: datetime, context: Dict[str, Any]) -> List[str]:
        """
        可选的候选标的二次过滤（默认返回原列表）
        """
        return candidates
