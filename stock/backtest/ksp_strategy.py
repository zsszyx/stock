import backtrader as bt
import pandas as pd
import numpy as np
import json
from datetime import datetime
from stock.strategy.base import BaseStrategy

class KSPStrategy(bt.Strategy):
    """
    KSP策略适配器 (Backtrader版)
    - 采用策略模式：具体买卖决策委派给 params.core_strategy
    - 兼容原有 KSP 行为，但核心逻辑已解耦
    """
    params = (
        ('core_strategy', None), # 必须是 BaseStrategy 的子类实例
        ('slots', 9),
        ('ksp_period', 5), # KSP 排名周期: 5, 7, 10, 14
        ('start_date', None), # 策略正式开始交易的日期 (str: YYYY-MM-DD)
        ('log_file', 'backtest_detailed_log.json'),
    )

    def __init__(self):
        self.trade_records = []
        self.daily_records = []
        self.trade_costs = {}
        self.to_sell_queue = {} # 记录待卖出的标的及其原因: {code: reason}
        
        # 确定排名列名
        self.rank_attr = f'ksp_sum_{self.p.ksp_period}d_rank'
        
        if self.p.core_strategy is None:
            raise ValueError("Must provide 'core_strategy' parameter (instance of BaseStrategy)")

    def _get_data(self, code):
        """Safe wrapper to get data by name, avoiding KeyError"""
        try:
            return self.getdatabyname(code)
        except (KeyError, IndexError):
            return None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                dt_str = self.data.datetime.date(0).strftime('%Y-%m-%d')
                self.trade_records.append({
                    'date': dt_str, 'action': 'BUY_FILL', 'code': order.data._name,
                    'price': float(order.executed.price), 'size': int(order.executed.size),
                    'cost': float(order.executed.value), 'comm': float(order.executed.comm)
                })
                self.trade_costs[order.data._name] = {
                    'avg_cost': float(order.executed.price), 
                    'date': dt_str,
                    'size': int(order.executed.size)
                }
            elif order.issell():
                dt_str = self.data.datetime.date(0).strftime('%Y-%m-%d')
                reason = getattr(order, 'sell_reason', 'unknown')
                self.trade_records.append({
                    'date': dt_str, 'action': 'SELL_FILL', 'code': order.data._name,
                    'price': float(order.executed.price), 'size': int(order.executed.size),
                    'cost': float(order.executed.value), 'comm': float(order.executed.comm),
                    'reason': reason
                })
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            dt_str = self.data.datetime.date(0).strftime('%Y-%m-%d')
            self.trade_records.append({
                'date': dt_str, 'action': 'ORDER_FAILED', 'code': order.data._name,
                'status': order.getstatusname()
            })

    def next(self):
        dt = self.data.datetime.date(0)
        dt_str = dt.strftime('%Y-%m-%d')
        dt_datetime = datetime.combine(dt, datetime.min.time())
        core = self.p.core_strategy
        
        # 判断是否在预热期 (如果当前日期 < 指定开始日期，则不执行任何交易逻辑)
        if self.p.start_date and dt_str < self.p.start_date:
            return

        # 0. 撤销所有尚未成交的买单 (实现：第二天买不上就空着，不留旧单)
        open_orders = self.broker.get_orders_open()
        for order in open_orders:
            if order.isbuy():
                self.cancel(order)

        # 1. 准备排名上下文
        rank_map = {}
        rank_5d_map = {}
        rank_10d_map = {}
        
        for d in self.datas:
            if d._name.startswith('_'): continue
            try:
                if len(d) > 0:
                    rank_map[d._name] = getattr(d.lines, self.rank_attr)[0]
                    rank_5d_map[d._name] = d.lines.ksp_sum_5d_rank[0]
                    rank_10d_map[d._name] = d.lines.ksp_sum_10d_rank[0]
            except Exception:
                continue

        sold_today = set() 

        # 2. 处理待卖出队列
        for code in list(self.to_sell_queue.keys()):
            data = self._get_data(code)
            if data is None or len(data) == 0:
                del self.to_sell_queue[code]
                continue
            
            if data.volume[0] <= 0:
                continue
            
            if len(data) > 1 and data.open[0] <= data.close[-1] * 0.905:
                continue
            
            pos = self.getposition(data)
            if pos.size > 0:
                reason = self.to_sell_queue[code]
                order = self.close(data=data)
                if order:
                    order.sell_reason = reason
                sold_today.add(code) 
            del self.to_sell_queue[code]

        # 3. 产生新的卖出决策
        for code in list(self.trade_costs.keys()):
            if code in self.to_sell_queue: continue
            
            data = self._get_data(code)
            if data is None or len(data) == 0: continue
            pos = self.getposition(data)
            if pos.size <= 0: continue
            
            buy_price = self.trade_costs[code]['avg_cost']
            current_close = data.close[0]
            context = {
                'rank': rank_map.get(code), 
                'rank_5d': rank_5d_map.get(code),
                'rank_10d': rank_10d_map.get(code),
                'ksp_sum_5d_rank': rank_5d_map.get(code),
                'ksp_sum_10d_rank': rank_10d_map.get(code),
                'open': data.open[0], 
                'close': data.close[0]
            }
            
            exit_reason = core.should_exit(code, buy_price, current_close, dt_datetime, context)
            if exit_reason:
                self.to_sell_queue[code] = exit_reason

        # 4. 买入决策
        active_positions = [p for p in self.datas if self.getposition(p).size > 0]
        position_count = len(active_positions)
        
        if position_count < self.p.slots:
            filter_ctx = {
                'rank_map': rank_map,
                'rank_5d_map': rank_5d_map,
                'rank_10d_map': rank_10d_map,
                'stock_data_map': {}
            }
            
            target_codes = core.select_targets(dt_datetime, filter_ctx)
                
            if target_codes:
                potential_codes = []
                for code in target_codes:
                    if (code in self.trade_costs or 
                        code in sold_today or 
                        code in self.to_sell_queue):
                        continue
                    
                    data = self._get_data(code)
                    if data is None or len(data) < 1: continue
                    
                    if data.volume[0] <= 0 or data.close[0] <= 0:
                        continue
                    
                    if len(data) > 1 and data.open[0] >= data.close[-1] * 1.095:
                        continue
                        
                    # 计算指标
                    ma_window = 20
                    available = len(data)
                    closes = data.close.get(ago=0, size=min(available, ma_window))
                    ma20 = np.mean(closes) if len(closes) > 0 else None
                    
                    vol_window = 5
                    vols = data.volume.get(ago=0, size=min(available, vol_window + 1))
                    if len(vols) > 1:
                        avg_vol = np.mean(vols[1:])
                        vol_ratio = vols[0] / avg_vol if avg_vol > 0 else 1.0
                    else:
                        vol_ratio = 1.0

                    filter_ctx['stock_data_map'][code] = {
                        'open': data.open[0],
                        'high': data.high[0],
                        'low': data.low[0],
                        'close': data.close[0],
                        'poc': data.poc[0] if hasattr(data, 'poc') else None,
                        'vol_ratio': vol_ratio,
                        'ma_20': ma20,
                        'rank': rank_map.get(code),
                        'ksp_sum_5d_rank': rank_5d_map.get(code)
                    }
                    potential_codes.append(code)

                passed_candidates = core.filter_candidates(potential_codes, dt_datetime, filter_ctx)
                
                target_pct = 1.0 / self.p.slots
                for code in passed_candidates:
                    if position_count >= self.p.slots: break
                    data = self._get_data(code)
                    self.order_target_percent(data=data, target=target_pct, exectype=bt.Order.Market)
                    position_count += 1

        self._log_daily_status(dt_str, rank_map)

    def _log_daily_status(self, dt_str, rank_map=None):
        cash = self.broker.getcash()
        positions_list = []
        position_value = 0.0
        for data in self.datas:
            if data._name.startswith('_'): continue
            pos = self.getposition(data)
            if pos.size > 0:
                code = data._name
                price = data.close[0]
                cost_info = self.trade_costs.get(code, {'avg_cost': price})
                val = float(pos.size * price)
                position_value += val
                positions_list.append({
                    'code': code, 'size': int(pos.size),
                    'buy_price': float(cost_info['avg_cost']),
                    'price': float(price),
                    'value': val,
                    'profit_pct': float((price - cost_info['avg_cost']) / cost_info['avg_cost']) if cost_info['avg_cost'] > 0 else 0.0,
                    'ksp_rank': float(rank_map.get(code, 999)) if rank_map else 999
                })
        total_value = cash + position_value
        active_codes = [p['code'] for p in positions_list]
        for code in list(self.trade_costs.keys()):
            if code not in active_codes:
                del self.trade_costs[code]
        self.daily_records.append({
            'date': dt_str, 'total_value': float(total_value),
            'cash': float(cash), 'position_value': float(position_value),
            'position_count': len(positions_list), 'positions': positions_list,
        })

    def stop(self):
        log_data = {
            'summary': {
                'final_value': float(self.broker.getvalue()),
                'total_trades': len(self.trade_records),
                'total_days': len(self.daily_records),
            },
            'daily_records': self.daily_records,
            'trade_records': self.trade_records,
        }
        try:
            with open(self.p.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
