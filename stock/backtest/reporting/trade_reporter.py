"""
交易报告生成器模块

功能:
- 分析每日买卖操作
- 计算每笔交易的盈亏
- 生成格式化的Markdown报告
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
        
    def analyze_trades(self, daily_records: List[Dict]) -> pd.DataFrame:
        """
        分析所有交易，提取买卖记录和盈亏
        
        Args:
            daily_records: 每日持仓记录列表
            
        Returns:
            交易记录DataFrame
        """
        # 追踪持仓历史: code -> [(buy_date, buy_price, buy_size), ...]
        position_history = defaultdict(list)
        closed_trades = []
        
        for i, day in enumerate(daily_records):
            date = day['date']
            positions = {p['code']: p for p in day.get('positions', [])}
            
            # 找出卖出的股票
            if i > 0:
                prev_positions = {p['code']: p for p in daily_records[i-1].get('positions', [])}
                for code, prev_pos in prev_positions.items():
                    if code not in positions:
                        # 卖出！
                        buys = position_history.get(code, [])
                        if buys:
                            buy_record = buys.pop(0)
                            buy_date, buy_price, buy_size = buy_record
                            
                            sell_price = prev_pos['price']
                            profit = (sell_price - buy_price) * buy_size
                            profit_pct = (sell_price - buy_price) / buy_price * 100
                            holding_days = (pd.to_datetime(date) - pd.to_datetime(buy_date)).days
                            
                            closed_trades.append({
                                'code': code,
                                'buy_date': buy_date,
                                'buy_price': round(buy_price, 2),
                                'buy_size': buy_size,
                                'sell_date': date,
                                'sell_price': round(sell_price, 2),
                                'profit': round(profit, 2),
                                'profit_pct': round(profit_pct, 2),
                                'holding_days': holding_days
                            })
            
            # 找出新增的买入
            if i > 0:
                prev_positions = {p['code']: p for p in daily_records[i-1].get('positions', [])}
                for code, pos in positions.items():
                    if code not in prev_positions:
                        position_history[code].append((date, pos['price'], pos['size']))
            else:
                for code, pos in positions.items():
                    position_history[code].append((date, pos['price'], pos['size']))
        
        return pd.DataFrame(closed_trades)
    
    def get_statistics(self, trades_df: pd.DataFrame) -> Dict:
        """获取交易统计"""
        if trades_df.empty:
            return {}
        
        total = len(trades_df)
        wins = trades_df[trades_df['profit'] > 0]
        losses = trades_df[trades_df['profit'] <= 0]
        
        return {
            'total_trades': total,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / total * 100 if total > 0 else 0,
            'total_profit': trades_df['profit'].sum(),
            'avg_profit': wins['profit'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['profit'].mean() if len(losses) > 0 else 0,
            'profit_ratio': abs(wins['profit'].mean() / losses['profit'].mean()) if len(wins) > 0 and len(losses) > 0 else 0,
            'avg_holding_days': trades_df['holding_days'].mean()
        }
    
    def generate_daily_report(self, daily_records: List[Dict], output_path: str):
        """
        生成每日详细交易报告
        
        Args:
            daily_records: 每日持仓记录
            output_path: 输出文件路径
        """
        # 分析交易
        trades_df = self.analyze_trades(daily_records)
        stats = self.get_statistics(trades_df)
        
        # 生成Markdown
        md = self._generate_markdown(daily_records, trades_df, stats)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md)
        
        return output_path
    
    def _generate_markdown(self, daily_records, trades_df, stats) -> str:
        """生成Markdown报告"""
        
        # 头部
        md = f"""# 📊 每日交易记录与持仓变化报告

## 回测概况

| 项目 | 数值 |
|------|------|
| 回测期间 | {daily_records[0]['date']} ~ {daily_records[-1]['date']} |
| 初始资金 | {self.initial_capital:,.0f} 元 |
| 最终资产 | {daily_records[-1]['total_value']:,.0f} 元 |
| 总收益率 | {(daily_records[-1]['total_value']/self.initial_capital-1)*100:+.2f}% |

## 交易统计

| 指标 | 数值 |
|------|------|
| 总交易次数 | {stats.get('total_trades', 0)} |
| 盈利交易 | {stats.get('winning_trades', 0)} ({stats.get('win_rate', 0):.1f}%) |
| 亏损交易 | {stats.get('losing_trades', 0)} |
| 总盈亏 | {stats.get('total_profit', 0):+,.0f} 元 |
| 平均盈利 | {stats.get('avg_profit', 0):+,.0f} 元 |
| 平均亏损 | {stats.get('avg_loss', 0):,.0f} 元 |
| 盈亏比 | {stats.get('profit_ratio', 0):.2f} : 1 |
| 平均持仓天数 | {stats.get('avg_holding_days', 0):.1f} 天 |

---

## 每日详细记录

"""
        
        # 追踪每日买卖
        position_history = defaultdict(list)
        
        for i, day in enumerate(daily_records):
            date = day['date']
            total_value = day['total_value']
            cash = day['cash']
            position_value = day['position_value']
            positions = {p['code']: p for p in day.get('positions', [])}
            
            # 当日涨跌
            if i > 0:
                prev_value = daily_records[i-1]['total_value']
                change = (total_value - prev_value) / prev_value * 100
                change_str = f"{change:+.2f}%"
            else:
                change_str = "---"
            
            # 找出买卖操作
            buys = []
            sells = []
            
            if i > 0:
                prev_positions = {p['code']: p for p in daily_records[i-1].get('positions', [])}
                
                # 新买入
                for code, pos in positions.items():
                    if code not in prev_positions:
                        buys.append(pos)
                
                # 卖出 (并计算盈亏)
                for code, prev_pos in prev_positions.items():
                    if code not in positions:
                        # 查找买入记录
                        history = position_history.get(code, [])
                        if history:
                            buy_record = history.pop(0)
                            buy_date, buy_price, buy_size = buy_record
                            profit = (prev_pos['price'] - buy_price) * buy_size
                            profit_pct = (prev_pos['price'] - buy_price) / buy_price * 100
                            sells.append({
                                'code': code,
                                'buy_price': buy_price,
                                'sell_price': prev_pos['price'],
                                'profit': profit,
                                'profit_pct': profit_pct,
                                'holding_days': (pd.to_datetime(date) - pd.to_datetime(buy_date)).days
                            })
            else:
                # 第一天全部是买入
                buys = list(positions.values())
            
            # 更新持仓历史
            for code, pos in positions.items():
                if code not in position_history:
                    position_history[code] = []
                # 检查是否已有记录
                existing = [h for h in position_history[code] if h[0] == date]
                if not existing:
                    position_history[code].append((date, pos['price'], pos['size']))
            
            # 写入日期章节
            md += f"### 📅 {date}\n\n"
            md += f"**总资产**: {total_value:,.0f} 元 ({change_str}) | "
            md += f"**现金**: {cash:,.0f} 元 | "
            md += f"**持仓**: {position_value:,.0f} 元 ({len(positions)}只)\n\n"
            
            # 买入记录
            if buys:
                md += f"**🟢 买入** ({len(buys)}只):\n\n"
                md += f"| 代码 | 数量 | 价格 | 市值 |\n"
                md += f"|------|------|------|------|\n"
                for b in buys:
                    md += f"| {b['code']} | {b['size']:,} | {b['price']:.2f} | {b['value']:,.0f} |\n"
                md += "\n"
            
            # 卖出记录
            if sells:
                md += f"**🔴 卖出** ({len(sells)}只):\n\n"
                md += f"| 代码 | 买入价 | 卖出价 | 盈亏 | 盈亏率 | 持仓天数 |\n"
                md += f"|------|--------|--------|------|--------|----------|\n"
                for s in sells:
                    profit_emoji = "🟢" if s['profit'] > 0 else "🔴"
                    md += f"| {s['code']} | {s['buy_price']:.2f} | {s['sell_price']:.2f} | {profit_emoji}{s['profit']:+,.0f} | {s['profit_pct']:+.1f}% | {s['holding_days']}天 |\n"
                md += "\n"
            
            # 当前持仓
            if positions:
                md += f"**📦 持仓明细** ({len(positions)}只):\n\n"
                md += f"| 代码 | 数量 | 价格 | 市值 | KSP排名 |\n"
                md += f"|------|------|------|------|---------|\n"
                for pos in sorted(positions.values(), key=lambda x: x['value'], reverse=True):
                    md += f"| {pos['code']} | {pos['size']:,} | {pos['price']:.2f} | {pos['value']:,.0f} | {int(pos.get('ksp_rank', 0))} |\n"
                md += "\n"
            
            md += "---\n\n"
        
        return md


def generate_trading_report(daily_records: List[Dict], output_path: str, initial_capital: float = 1000000):
    """
    生成交易报告的便捷函数
    
    Args:
        daily_records: 每日持仓记录
        output_path: 输出文件路径
        initial_capital: 初始资金
    """
    analyzer = TradeAnalyzer(initial_capital)
    return analyzer.generate_daily_report(daily_records, output_path)


if __name__ == "__main__":
    # 测试
    with open('../backtest_detailed_log.json', 'r') as f:
        data = json.load(f)
    
    generate_trading_report(
        data['daily_records'], 
        './test_report.md',
        1000000
    )
    print("报告已生成: test_report.md")
