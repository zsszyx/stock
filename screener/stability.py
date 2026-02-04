import pandas as pd
import numpy as np
from stock.screener.base import BaseScreener
from stock.data.context import MarketContext

class StabilityScreener(BaseScreener):
    """
    稳定性选股器：筛选最近 N 天内横盘振荡的股票。
    逻辑：使用 Context 中的原始数据计算过去 N 天的最大涨跌幅。
    """
    def __init__(self, window_days: int = 5, threshold: float = 0.05):
        super().__init__()
        self.window_days = window_days
        self.threshold = threshold

    def scan(self, context: MarketContext) -> pd.DataFrame:
        """
        通过分钟级上下文获取窗口数据。
        """
        # 确定需要计算的股票范围
        codes = context.candidate_codes if self.filter_candidates else None
        
        # 使用专用的五分钟上下文接口，获取指定窗口的数据
        df = context.minutes5.get_window(date=context.current_date, 
                                          window_days=self.window_days,
                                          codes=codes)
        
        if df.empty:
            return pd.DataFrame()

        # 2. 按股票分组计算最大/最小价格和基准价
        results = []
        grouped = df.groupby('code')
        
        target_date_str = context.current_date.strftime('%Y-%m-%d')

        for code, group in grouped:
            # 确保按时间排序
            sorted_group = group.sort_values(['date', 'time'])
            
            # 获取所有收盘价
            closes = sorted_group['close'].values
            if len(closes) == 0:
                continue
                
            hi = np.max(closes)
            lo = np.min(closes)
            
            # 获取当前交易日的收盘价作为参考基准
            current_day_data = sorted_group[sorted_group['date'] == target_date_str]
            if current_day_data.empty:
                # 如果当前交易日没数据，跳过
                continue
            ref_price = current_day_data['close'].iloc[-1]

            # 计算相对于当前价格的最大涨幅和最大跌幅
            max_gain = (hi - ref_price) / ref_price
            max_drop = (lo - ref_price) / ref_price

            if max_gain <= self.threshold and max_drop >= -self.threshold:
                results.append({'code': code})

        final_df = pd.DataFrame(results)
        
        print(f"[{target_date_str}] StabilityScreener (Context Mode): {len(grouped)} -> {len(final_df)} stocks passed.")
        
        return final_df
