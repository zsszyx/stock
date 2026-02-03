from stock.backtest.strategy import Strategy
from stock.backtest.models import Direction
from stock.sql_op.op import SqlOp
from stock.sql_op import sql_config
import pandas as pd
import numpy as np
from datetime import timedelta

class SingularityBacktestStrategy(Strategy):
    """
    【黄金三角】爆发突破策略
    核心逻辑：能量(K) - 方向(S) - 触发(Price action)
    1. 方向：P形筹码 (Skew < -1.0) -> 主力高位吸筹，拒绝低价。
    2. 能量：能量压缩 (Kurt > 1.0 且 飙升) -> 筹码共识，暴风雨前的宁静。
    3. 触发：价格稳住 (Close >= POC 且 缩量/低波) -> 点火即燃。
    """
    def initialize(self):
        print("Strategy: Golden Triangle Singularity (P-Shape + High Energy)")
        self.sql_operator = SqlOp()
        self.last_date = None
        self.candidates = set() # Set of codes to buy today
        self.holdings = {} # code -> {entry_date, entry_price}
        self.hold_days = 5 # 持仓周期
        
    def calculate_poc(self, df: pd.DataFrame, price_bins: int = 50) -> float:
        """Calculates POC (Point of Control) - Price level with max volume."""
        if 'avg_price' not in df.columns:
            df = df.copy()
            # Avoid division by zero
            df['avg_price'] = np.where(df['volume'] > 0, df['amount'] / df['volume'], df['close'])
            
        hist, bin_edges = np.histogram(
            df['avg_price'], 
            bins=price_bins, 
            weights=df['volume'],
            range=(df['avg_price'].min(), df['avg_price'].max())
        )
        
        if len(hist) == 0:
            return 0.0
            
        max_vol_idx = np.argmax(hist)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers[max_vol_idx]

    def check_singularity_signal(self, target_date: str, active_codes: list):
        """
        Checks for the 'Golden Triangle' setup on target_date.
        """
        # 1. 基础筛选：方向(S < -1.0) + 能量(K > 1.0) + 板块(主板/创业板)
        # S < -1.0: 严格的 P形
        # K > 1.0: 至少是尖峰分布
        stats_query = f"""
            SELECT code, weighted_skew, weighted_kurt
            FROM {sql_config.daily_stats_table_name}
            WHERE strftime('%Y-%m-%d', date) = '{target_date}'
              AND weighted_skew < -1.0 
              AND weighted_kurt > 1.0
              AND (code LIKE '60%' OR code LIKE '00%' OR code LIKE '30%')
        """
        candidates_df = self.sql_operator.query(stats_query)
        
        if candidates_df is None or candidates_df.empty:
            return []
            
        # Filter strictly for codes that are "active" in our backtest universe
        # Normalize active_codes to 6 digits for comparison
        active_codes_norm = [c[-6:] for c in active_codes]
        candidates_df = candidates_df[candidates_df['code'].isin(active_codes_norm)]
        
        if candidates_df.empty:
            return []
            
        # Map back to full codes (sh.xxxxxx)
        code_map = {c[-6:]: c for c in active_codes}
        
        # 2. 能量飙升确认：K值必须上升 (K_today > K_prev)
        prev_date_query = f"""
            SELECT MAX(date) FROM {sql_config.daily_stats_table_name} 
            WHERE strftime('%Y-%m-%d', date) < '{target_date}'
        """
        res_prev = self.sql_operator.query(prev_date_query)
        if res_prev is None or res_prev.empty or res_prev.iloc[0, 0] is None:
            return []
            
        prev_date = pd.to_datetime(res_prev.iloc[0, 0]).strftime('%Y-%m-%d')
        
        prev_stats_query = f"""
            SELECT code, weighted_kurt as prev_kurt
            FROM {sql_config.daily_stats_table_name}
            WHERE strftime('%Y-%m-%d', date) = '{prev_date}'
        """
        prev_stats_df = self.sql_operator.query(prev_stats_query)
        
        if prev_stats_df is not None and not prev_stats_df.empty:
            candidates_df = pd.merge(candidates_df, prev_stats_df, on='code', how='inner')
            # Criteria: K Rising
            candidates_df = candidates_df[candidates_df['weighted_kurt'] > candidates_df['prev_kurt']]
        else:
            return []
            
        if candidates_df.empty:
            return []

        final_candidates = []
        
        # 3. 触发点确认：K线形态 + POC位置
        for code_short in candidates_df['code']:
            full_code = code_map.get(code_short)
            if not full_code:
                continue
                
            # Fetch 5-min data for the target date
            k_data = self.sql_operator.read_k_data_by_date_range(sql_config.mintues5_table_name, target_date, target_date)
            
            if k_data.empty:
                continue
            
            # Filter for specific code (Safe copy)
            if '.' in k_data['code'].iloc[0]:
                k_data_code = k_data[k_data['code'] == full_code].copy()
            else:
                k_data_code = k_data[k_data['code'] == full_code].copy()
                
            if k_data_code.empty:
                continue
            
            # Numeric conversion
            cols_to_numeric = ['open', 'high', 'low', 'close', 'amount', 'volume']
            for col in cols_to_numeric:
                k_data_code[col] = pd.to_numeric(k_data_code[col], errors='coerce')

            # Filter volume > 0
            k_data_code = k_data_code[k_data_code['volume'] > 0].copy()
            if k_data_code.empty:
                continue
                
            poc = self.calculate_poc(k_data_code)
            if poc == 0:
                continue
                
            daily_close = k_data_code['close'].iloc[-1]
            daily_high = k_data_code['high'].max()
            daily_low = k_data_code['low'].min()
            
            # --- 核心过滤逻辑 ---
            
            # 1. 价格稳住 POC：Close >= POC
            # 并且不能偏离太远 (例如不超过 2%)，保证是"蓄势"而不是"已经飞了"
            if daily_close < poc:
                continue # 跌破 POC，形态走坏
                
            if daily_close > poc * 1.02:
                continue # 已经启动，错过最佳埋伏点
                
            # 2. 波动率极低 (压缩)：振幅 < 3%
            # (High - Low) / Close
            amplitude = (daily_high - daily_low) / daily_close
            if amplitude > 0.03:
                continue # 波动太大，分歧还不够小
                
            final_candidates.append(full_code)
            
        return final_candidates

    def next(self, bars):
        if not bars:
            return
            
        current_time = list(bars.values())[0].time
        current_date = current_time.strftime('%Y-%m-%d')
        
        # --- 日期变更逻辑 ---
        if self.last_date != current_date:
            if self.last_date is not None:
                # 1. 卖出逻辑 (持有到期)
                to_sell = []
                for code, pos_info in self.holdings.items():
                    entry_date = pos_info['entry_date']
                    days_held = (pd.to_datetime(current_date) - pd.to_datetime(entry_date)).days
                    if days_held >= self.hold_days:
                        to_sell.append(code)
                
                for code in to_sell:
                    pos = self.broker.get_position(code)
                    if pos.quantity > 0:
                        print(f"  [Sell] {code} (Expired {self.hold_days} days)")
                        self.sell(code, pos.quantity)
                        del self.holdings[code]

                # 2. 选股逻辑 (基于昨日数据)
                # 使用当前数据流中存在的 code 作为 universe proxy
                active_codes = list(self.data_history.keys())
                
                found = self.check_singularity_signal(self.last_date, active_codes)
                if found:
                    print(f"\n[Signal] Golden Triangle candidates on {self.last_date}: {found}")
                    for code in found:
                        self.candidates.add(code)
            
            self.last_date = current_date

        # --- 盘中交易 ---
        for code in list(self.candidates):
            if code in bars:
                if code not in self.holdings:
                    price = bars[code].close
                    # 资金管理：每只股票投入 50,000 元 (假设总资金足够)
                    target_value = 50000 
                    quantity = int(target_value / price / 100) * 100
                    
                    if quantity > 0:
                        print(f"  [Buy] {code} @ {price}, Qty: {quantity}")
                        self.buy(code, quantity)
                        self.holdings[code] = {
                            'entry_date': current_date,
                            'entry_price': price
                        }
                    
                self.candidates.remove(code)
