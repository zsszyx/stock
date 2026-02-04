import pandas as pd
from typing import List, Optional
from datetime import datetime

from .base import BaseScreener
from stock.data.context import MarketContext

class ScreenerPipeline:
    """
    选股流水线，负责逻辑串联。
    """
    def __init__(self, screeners: List[BaseScreener]):
        self.screeners = screeners

    def run(self, date: datetime, context: MarketContext) -> pd.DataFrame:
        """
        运行选股流程。
        """
        # 设置当前日期，供 Screener 内部引用
        context.current_date = date
        context.candidate_codes = None
        
        final_picks = pd.DataFrame()
        
        # 顺序执行选股器
        for i, screener in enumerate(self.screeners):
            picks = screener.scan(context)
            if picks is None or picks.empty:
                context.candidate_codes = []
                return pd.DataFrame()
            
            # 更新上下文中的候选名单，供下一个 Screener 使用
            context.candidate_codes = picks['code'].tolist()
            final_picks = picks

        return final_picks