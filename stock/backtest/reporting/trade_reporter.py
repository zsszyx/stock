"""
交易报告生成器模块 (审计增强版)

功能:
- 基于真实成交记录 (trade_records) 分析盈亏
- 消除收盘价与次日开盘价之间的跳空误差
- 生成 Markdown 格式的详细审计报告
"""

import json
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict


class TradeAnalyzer:
    """交易记录分析器"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        
    def analyze_trades_v2(self, trade_records: List[Dict]) -> pd.DataFrame:
        """
        根据原始交易记录分析盈亏，确保价格准确
        """
        closed_trades = []
        # code -> list of {'date', 'price', 'size'}
        active_buys = defaultdict(list)
        
        for rec in trade_records:
            code = rec['code']
            if rec['action'] == 'BUY_FILL':
                active_buys[code].append({
                    'date': rec['date'],
                    'price': rec['price'],
                    'size': rec['size']
                })
            elif rec['action'] == 'SELL_FILL':
                buys = active_buys.get(code, [])
                if buys:
                    buy_rec = buys.pop(0)
                    sell_price = rec['price']
                    buy_price = buy_rec['price']
                    size = abs(rec['size']) 
                    
                    profit = (sell_price - buy_price) * size
                    profit_pct = (sell_price - buy_price) / buy_price * 100 if buy_price > 0 else 0
                    holding_days = (pd.to_datetime(rec['date']) - pd.to_datetime(buy_rec['date'])).days
                    
                    closed_trades.append({
                        'code': code,
                        'buy_date': buy_rec['date'],
                        'buy_price': round(buy_price, 2),
                        'buy_size': size,
                        'sell_date': rec['date'],
                        'sell_price': round(sell_price, 2),
                        'profit': round(profit, 2),
                        'profit_pct': round(profit_pct, 2),
                        'holding_days': holding_days
                    })
        
        return pd.DataFrame(closed_trades)

    def get_statistics(self, trades_df: pd.DataFrame) -> Dict:
        """获取交易统计"""
        if trades_df.empty:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0, 'total_profit': 0, 'avg_profit': 0,
                'avg_loss': 0, 'profit_ratio': 0, 'avg_holding_days': 0
            }
        
        total = len(trades_df)
        wins = trades_df[trades_df['profit'] > 0]
        losses = trades_df[trades_df['profit'] <= 0]
        
        avg_profit = wins['profit'].mean() if len(wins) > 0 else 0
        avg_loss = losses['profit'].mean() if len(losses) > 0 else 0
        profit_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        
        return {
            'total_trades': total,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / total * 100 if total > 0 else 0,
            'total_profit': trades_df['profit'].sum(),
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_ratio': profit_ratio,
            'avg_holding_days': trades_df['holding_days'].mean()
        }
    
    def generate_detailed_report(self, log_data: Dict, output_path: str):
        """生成审计报告"""
        daily_records = log_data.get('daily_records', [])
        trade_records = log_data.get('trade_records', [])
        
        trades_df = self.analyze_trades_v2(trade_records)
        stats = self.get_statistics(trades_df)
        
        md = self._generate_markdown(daily_records, trade_records, trades_df, stats)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md)
        return output_path

    def _generate_markdown(self, daily_records, trade_records, trades_df, stats) -> str:
        md = f"""# 📊 策略回测审计报告 (Corrected v2)

## 1. 回测概况

| 项目 | 数值 |
|------|------|
| 回测期间 | {daily_records[0]['date']} ~ {daily_records[-1]['date']} |
| 初始资金 | {self.initial_capital:,.0f} 元 |
| 最终资产 | {daily_records[-1]['total_value']:,.0f} 元 |
| 总收益率 | {(daily_records[-1]['total_value']/self.initial_capital-1)*100:+.2f}% |

## 2. 核心交易统计 (基于实际成交价)

| 指标 | 数值 |
|------|------|
| 总已平仓次数 | {stats['total_trades']} |
| 盈利次数 | {stats['winning_trades']} ({stats['win_rate']:.1f}%) |
| 亏损次数 | {stats['losing_trades']} |
| 总盈亏金额 | {stats['total_profit']:+,.0f} 元 |
| 平均盈利 | {stats['avg_profit']:+,.0f} 元 |
| 平均亏损 | {stats['avg_loss']:+,.0f} 元 |
| 盈亏比 | {stats['profit_ratio']:.2f} : 1 |
| 平均持仓时长 | {stats['avg_holding_days']:.1f} 天 |

---

## 3. 详细成交历史

"""
        if not trades_df.empty:
            md += "| 代码 | 买入日期 | 买入价 | 卖出日期 | 卖出价 | 盈亏 | 盈亏率 | 持仓天数 |\n"
            md += "|------|----------|--------|----------|--------|------|--------|----------|\n"
            for _, row in trades_df.sort_values('sell_date').iterrows():
                profit_emoji = "🟢" if row['profit'] > 0 else "🔴"
                md += f"| {row['code']} | {row['buy_date']} | {row['buy_price']:.2f} | {row['sell_date']} | {row['sell_price']:.2f} | {profit_emoji}{row['profit']:+,.0f} | {row['profit_pct']:+.1f}% | {row['holding_days']} |\n"
        else:
            md += "无已平仓交易记录。\n"

        md += "\n---\n\n## 4. 每日资产摘要 (抽样)\n\n"
        md += "| 日期 | 总资产 | 现金 | 持仓市值 | 持仓数 | 当日涨跌 |\n"
        md += "|------|--------|------|----------|--------|----------|\n"
        
        for i, day in enumerate(daily_records):
            if i % 10 == 0 or i == len(daily_records) - 1:
                change = ""
                if i > 0:
                    prev_val = daily_records[i-1]['total_value']
                    change = f"{(day['total_value']/prev_val-1)*100:+.2f}%"
                md += f"| {day['date']} | {day['total_value']:,.0f} | {day['cash']:,.0f} | {day['position_value']:,.0f} | {day['position_count']} | {change} |\n"

        return md


def generate_trading_report(log_data: Dict, output_path: str, initial_capital: float = 1000000):
    analyzer = TradeAnalyzer(initial_capital)
    return analyzer.generate_detailed_report(log_data, output_path)
