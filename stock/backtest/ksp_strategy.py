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
        ('log_file', 'backtest_detailed_log.json'),
    )

    def __init__(self):
        self.trade_records = []
        self.daily_records = []
        self.trade_costs = {}
        self.pending_orders = {}
        
        if self.p.core_strategy is None:
            raise ValueError("Must provide 'core_strategy' parameter (instance of BaseStrategy)")

    def next(self):
        dt = self.data.datetime.date(0)
        dt_str = dt.strftime('%Y-%m-%d')
        dt_datetime = datetime.combine(dt, datetime.min.time())
        core = self.p.core_strategy
        
        # 1. 准备当日排名上下文
        rank_map = {}
        for d in self.datas:
            if hasattr(d, 'ksp_sum_5d_rank'):
                rank_map[d._name] = d.ksp_sum_5d_rank[0]

        # 2. 获取当日买入信号
        target_codes = core.select_targets(dt_datetime, {'rank_map': rank_map})
        target_codes = target_codes if target_codes else []
        sold_today = set()

        # 3. 严格执行退出逻辑 (移除持有保护，达标即卖)
        for code in list(self.trade_costs.keys()):
            data = self.getdatabyname(code)
            if data is None: continue
            pos = self.getposition(data)
            if pos.size <= 0: continue
            
            buy_price = self.trade_costs[code]['avg_cost']
            current_close = data.close[0]
            
            context = {
                'rank': rank_map.get(code),
                'open': data.open[0],
                'close': data.close[0]
            }
            
            # 决策：是否满足任意退出条件？
            exit_reason = core.should_exit(code, buy_price, current_close, dt_datetime, context)
            
            if exit_reason:
                # 达标即平仓，空出槽位给更强的标的
                self.close(data=data)
                self.trade_records.append({
                    'date': dt_str, 'action': 'SELL_SIGNAL', 'code': code,
                    'price': float(data.close[0]), 'size': int(pos.size),
                    'profit_pct': float((data.close[0] - buy_price) / buy_price),
                    'reason': exit_reason,
                })
                del self.trade_costs[code]
                sold_today.add(code) # 拦截当日重买

        # 4. 买入决策 (新鲜血液轮动)
        position_count = len([c for c, cost in self.trade_costs.items() 
                             if self.getposition(self.getdatabyname(c)).size > 0])
        
        if position_count >= self.p.slots:
            self._log_daily_status(dt_str)
            return
            
        target_pct = 1.0 / self.p.slots
        
        for code in target_codes:
            if position_count >= self.p.slots: break
            if code in self.trade_costs or code in sold_today: continue 
            
            data = self.getdatabyname(code)
            if data is None or len(data) <= 1: continue
            
            context = {
                'rank_map': rank_map, 'poc': data.poc[0] if hasattr(data, 'poc') else None,
                'open': data.open[0], 'rank': rank_map.get(code)
            }
            
            if not core.filter_candidates([code], dt_datetime, context): continue
            exec_price = core.get_execution_price(code, dt_datetime, context)
            if exec_price <= 0.01: continue
            
            self.order_target_percent(data=data, target=target_pct,
                                     exectype=bt.Order.Limit, price=exec_price)
            
            self.trade_records.append({
                'date': dt_str, 'action': 'BUY_SIGNAL', 'code': code,
                'price': float(exec_price), 'target_pct': float(target_pct),
                'ksp_rank': float(context['rank']) if context['rank'] is not None else 999,
            })
            self.trade_costs[code] = {'avg_cost': exec_price, 'date': dt_str}
            position_count += 1

        self._log_daily_status(dt_str)
        
        if position_count >= self.p.slots:
            self._log_daily_status(dt_str)
            return
            
        target_pct = 1.0 / self.p.slots
        
        for code in target_codes:
            if position_count >= self.p.slots: break
            # 关键：如果在持有中，或者今天刚发出卖出信号(sold_today)，则不买入
            if code in self.trade_costs or code in sold_today: 
                continue 
            
            data = self.getdatabyname(code)
            if data is None or len(data) <= 1: continue
            
            context = {
                'rank_map': rank_map, 'poc': data.poc[0] if hasattr(data, 'poc') else None,
                'open': data.open[0], 'rank': rank_map.get(code)
            }
            
            if not core.filter_candidates([code], dt_datetime, context): continue
            exec_price = core.get_execution_price(code, dt_datetime, context)
            if exec_price <= 0.01: continue
            
            self.order_target_percent(data=data, target=target_pct,
                                     exectype=bt.Order.Limit, price=exec_price)
            
            self.trade_records.append({
                'date': dt_str, 'action': 'BUY_SIGNAL', 'code': code,
                'price': float(exec_price), 'target_pct': float(target_pct),
                'ksp_rank': float(context['rank']) if context['rank'] is not None else 999,
            })
            self.trade_costs[code] = {'avg_cost': exec_price, 'date': dt_str}
            position_count += 1

        self._log_daily_status(dt_str)

    def _log_daily_status(self, dt_str):
        cash = self.broker.getcash()
        total_value = self.broker.getvalue()
        
        positions_list = []
        for code, cost_info in self.trade_costs.items():
            data = self.getdatabyname(code)
            if data is None: continue
            pos = self.getposition(data)
            if pos.size <= 0: continue
            price = data.close[0]
            positions_list.append({
                'code': code, 'size': int(pos.size),
                'buy_price': float(cost_info['avg_cost']),
                'current_price': float(price),
                'value': float(pos.size * price),
                'profit_pct': float((price - cost_info['avg_cost']) / cost_info['avg_cost']) if cost_info['avg_cost'] > 0 else 0.0,
            })
        
        self.daily_records.append({
            'date': dt_str, 'total_value': float(total_value),
            'cash': float(cash), 'position_value': float(total_value - cash),
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
            print(f"\n✅ 详细日志已保存: {self.p.log_file}")
        except Exception as e:
            print(f"❌ 保存日志失败: {e}")


    def _log_daily_status(self, dt_str):
        cash = self.broker.getcash()
        total_value = self.broker.getvalue()
        
        positions_list = []
        for code, cost_info in self.trade_costs.items():
            data = self.getdatabyname(code)
            if data is None:
                continue
            pos = self.getposition(data)
            if pos.size <= 0:
                continue
            price = data.close[0]
            positions_list.append({
                'code': code, 'size': int(pos.size),
                'buy_price': float(cost_info['avg_cost']),
                'current_price': float(price),
                'value': float(pos.size * price),
                'profit_pct': float((price - cost_info['avg_cost']) / cost_info['avg_cost']) if cost_info['avg_cost'] > 0 else 0.0,
            })
        
        self.daily_records.append({
            'date': dt_str, 'total_value': float(total_value),
            'cash': float(cash), 'position_value': float(total_value - cash),
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
            print(f"\n✅ 详细日志已保存: {self.p.log_file}")
        except Exception as e:
            print(f"❌ 保存日志失败: {e}")
