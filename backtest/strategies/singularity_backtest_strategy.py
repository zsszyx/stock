from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, time, date

# Adjust import based on execution context
try:
    from stock.backtest.strategy import Strategy
    from stock.data.context import MarketContext
    from stock.backtest.models import Bar, Direction
    from stock.sql_op.op import SqlOp
    from stock.sql_op import sql_config
    from stock.screener.singularity import SingularityScreener
except ImportError:
    # Fallback if running from inside the package or different path
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from stock.backtest.strategy import Strategy
    from stock.data.context import MarketContext
    from stock.backtest.models import Bar, Direction
    from stock.sql_op.op import SqlOp
    from stock.sql_op import sql_config
    from stock.screener.singularity import SingularityScreener

class SingularityStrategy(Strategy):
    """
    Singularity Breakout Strategy
    """
    
    def __init__(self, broker, context: MarketContext, skew_threshold: float = -0.5, kurt_threshold: float = 1.0, top_n: int = 5):
        super().__init__(broker, context)
        self.top_n = top_n
        
        # 使用 Pipeline 架构
        from stock.screener.singularity import SingularityScreener
        from stock.screener.stability import StabilityScreener
        from stock.screener.pipeline import ScreenerPipeline
        
        # 步骤 1: 筛选最近 5 天横盘的股票 (最大涨跌幅 < 5%)
        stability_screener = StabilityScreener(window_days=5, threshold=0.05)
        # 步骤 2: 在稳定的股票中寻找奇异点 (只计算通过第一步筛选的股票)
        singularity_screener = SingularityScreener(skew_threshold=skew_threshold, top_n=top_n, filter_candidates=True)
        
        # 注意：这里的 Pipeline 不再负责加载数据，仅负责串联逻辑
        self.pipeline = ScreenerPipeline(screeners=[stability_screener, singularity_screener])
        
        self.intraday_data: Dict[str, List[Bar]] = {}
        self.pending_buys: List[Tuple[str, float, float]] = [] # (code, prev_close, allocated_cash)
        self.last_date: Optional[date] = None
        self.holdings_order: List[str] = [] # FIFO queue for holdings
        
    def initialize(self):
        print("SingularityStrategy Initialized.")

    def screen(self, start_date: str, end_date: str) -> List[str]:
        """扫描日期范围内的所有信号以确定 Universe"""
        print(f"Strategy: Pre-scanning for Universe [{start_date} to {end_date}]...")
        all_dates = self.context.minutes5.get_date_range(start_date, end_date)
        universe_codes = set()
        
        for date_str in all_dates:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=15, minute=0)
            picks = self.on_screen(date_obj)
            if not picks.empty:
                universe_codes.update(picks['code'].tolist())
        
        return list(universe_codes)

    def on_screen(self, date: datetime) -> pd.DataFrame:
        """执行每日选股流程"""
        # 这里的 Pipeline.run 现在直接基于 self.context 工作
        return self.pipeline.run(date, context=self.context)


    def next(self, bars: Dict[str, Bar]):
        # Check for new day
        current_bar = list(bars.values())[0] if bars else None
        if current_bar:
            if self.last_date is None or current_bar.time.date() > self.last_date:
                # New day detected! Process pending buys from yesterday's signals
                self._process_pending_buys(bars)
                self.last_date = current_bar.time.date()

        # 1. Accumulate Data
        for code, bar in bars.items():
            if code not in self.intraday_data:
                self.intraday_data[code] = []
            self.intraday_data[code].append(bar)
            
        # 2. Check for Market Close (15:00)
        if not bars:
            return
            
        current_time = list(bars.values())[0].time
        if current_time.time() == time(15, 0):
            self._rebalance_portfolio(current_time, bars)
            # Clear for next day
            self.intraday_data = {}

    def _process_pending_buys(self, first_bars_of_day: Dict[str, Bar]):
        for code, prev_close, allocated_cash in self.pending_buys:
            if code in first_bars_of_day:
                open_price = first_bars_of_day[code].open
                if open_price <= prev_close * 1.02:
                    qty = int(allocated_cash // open_price)
                    qty = (qty // 100) * 100
                    if qty > 0:
                        print(f"[{first_bars_of_day[code].time}] BUY {code} @ {open_price:.2f} (Next day open check passed: {open_price:.2f} <= {prev_close:.2f} * 1.02)")
                        self.buy(code, qty, open_price)
                        self.holdings_order.append(code)
                else:
                    print(f"[{first_bars_of_day[code].time}] SKIP BUY {code} (Open too high: {open_price:.2f} > {prev_close:.2f} * 1.02)")
        self.pending_buys = []

    def _rebalance_portfolio(self, current_time, current_bars):
        # 通过接口获取今日选股结果
        top_picks = self.on_screen(current_time)
        
        if top_picks.empty:
            return

        # 3. Generate Target Candidates (Close within 1% of POC)
        target_candidates = []
        for _, row in top_picks.iterrows():
            if row['close'] >= row['poc'] * 0.99 and row['close'] <= row['poc'] * 1.01:
                target_candidates.append(row['code'])
                
        # 4. Filter New Signals (not already held or pending)
        new_signals = [c for c in target_candidates 
                       if c not in self.holdings_order 
                       and c not in [pb[0] for pb in self.pending_buys]]
        
        if not new_signals:
            return

        # 5. Execute FIFO Rotation
        current_cash = self.broker.cash
        # Note: In a real broker, cash might only update after sell settles.
        # Here we assume immediate cash availability or enough margin.
        
        for code in new_signals:
            # Check if we need to make room
            if len(self.holdings_order) + len(self.pending_buys) >= self.top_n:
                if self.holdings_order:
                    to_sell = self.holdings_order.pop(0)
                    pos = self.broker.get_position(to_sell)
                    if pos and pos.quantity > 0:
                        price = current_bars.get(to_sell, self.intraday_data[to_sell][-1]).close
                        print(f"[{current_time}] FIFO SELL {to_sell} to make room for {code}")
                        self.sell(to_sell, pos.quantity, price)
                else:
                    # If somehow we have no holdings but are at capacity (all pending), 
                    # we skip this signal to avoid over-committing.
                    continue

            # Queue for tomorrow's open
            # We recalculate amount_per_trade for each to be simple, 
            # though usually it's cash / empty_slots.
            empty_slots = self.top_n - len(self.holdings_order) - len(self.pending_buys)
            if empty_slots > 0:
                amount_per_trade = self.broker.cash / empty_slots
                close_price = current_bars.get(code, self.intraday_data[code][-1]).close
                print(f"[{current_time}] SIGNAL: {code} selected. Pending buy at tomorrow's open (Ref Close: {close_price:.2f})")
                self.pending_buys.append((code, close_price, amount_per_trade))
