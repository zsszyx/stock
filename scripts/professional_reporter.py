import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def generate_full_report():
    # 设置路径
    daily_file = 'output/data/backtest_daily.csv'
    trades_file = 'output/data/backtest_trades.csv'
    plot_dir = 'output/plots'
    report_dir = 'output/reports'
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    if not os.path.exists(daily_file) or not os.path.exists(trades_file):
        print(f"❌ 缺失回测数据文件: {daily_file}")
        return

    # 1. 指标计算 (基于每日净值)
    df_daily = pd.read_csv(daily_file)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.sort_values('date')
    
    initial_cash = df_daily['total_value'].iloc[0]
    final_value = df_daily['total_value'].iloc[-1]
    total_return = (final_value / initial_cash) - 1
    
    days_delta = (df_daily['date'].max() - df_daily['date'].min()).days
    annual_return = (1 + total_return) ** (365 / days_delta) - 1 if days_delta > 0 else 0
    
    df_daily['daily_ret'] = df_daily['total_value'].pct_change()
    volatility = df_daily['daily_ret'].std() * np.sqrt(252)
    sharpe = (annual_return / volatility) if volatility > 0 else 0
    
    downside_ret = df_daily[df_daily['daily_ret'] < 0]['daily_ret']
    downside_vol = downside_ret.std() * np.sqrt(252)
    sortino = (annual_return / downside_vol) if downside_vol > 0 else 0
    
    df_daily['cum_max'] = df_daily['total_value'].cummax()
    df_daily['drawdown'] = (df_daily['total_value'] - df_daily['cum_max']) / df_daily['cum_max']
    max_drawdown = df_daily['drawdown'].min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # 2. 成交配对分析
    df_signals = pd.read_csv(trades_file)
    completed_trades = []
    active_buys = {}

    for _, row in df_signals.iterrows():
        code = row['code']
        action = row['action']
        if action == 'BUY_SIGNAL':
            active_buys[code] = row
        elif action == 'SELL_SIGNAL' and code in active_buys:
            buy_row = active_buys.pop(code)
            duration = (pd.to_datetime(row['date']) - pd.to_datetime(buy_row['date'])).days
            completed_trades.append({
                'code': code,
                'entry_date': buy_row['date'],
                'exit_date': row['date'],
                'entry_price': buy_row['price'],
                'exit_price': row['price'],
                'profit_pct': row['profit_pct'],
                'duration': duration,
                'reason': row['reason']
            })
    
    df_trades = pd.DataFrame(completed_trades)
    win_rate = (df_trades['profit_pct'] > 0).mean() if not df_trades.empty else 0
    
    # 3. 绘图
    plt.figure(figsize=(12, 7))
    plt.plot(df_daily['date'], df_daily['total_value']/initial_cash, label='Strategy Equity', color='blue')
    plt.title('Strategy Equity Curve (2025-2026)')
    plt.grid(True, alpha=0.3)
    chart_path = os.path.join(plot_dir, 'equity_curve.png')
    plt.savefig(chart_path)
    plt.close()

    # 4. 生成报告文本
    report_lines = [
        "# 📈 KSP 策略回测专业报告 (V6 纯粹形态版)",
        f"\n> **回测周期**: {df_daily['date'].min().date()} 至 {df_daily['date'].max().date()}",
        f"> **初始资金**: {initial_cash:,.2f} | **最终资产**: {final_value:,.2f}",
        "\n## 📊 核心指标",
        "| 指标 | 数值 |",
        "| :--- | :--- |",
        f"| 总收益率 | {total_return:.2%} |",
        f"| 年化收益率 | {annual_return:.2%} |",
        f"| 最大回撤 | {max_drawdown:.2%} |",
        f"| 夏普比率 | {sharpe:.2f} |",
        f"| 索提诺比率 | {sortino:.2f} |",
        f"| 卡尔玛比率 | {calmar:.2f} |",
        f"| 年化波动率 | {volatility:.2%} |",
        f"\n## 🏹 交易统计",
        f"| 统计项 | 数值 |",
        f"| :--- | :--- |",
        f"| 总成交笔数 | {len(df_trades)} |",
        f"| 胜率 | {win_rate:.2%} |",
        f"| 平均持仓天数 | {df_trades['duration'].mean():.1f} 天 |",
        "\n## 📜 最近 50 笔成交明细",
        "| 退出日期 | 代码 | 方向 | 盈亏 | 持仓时长 | 原因 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for _, row in df_trades.tail(50).iloc[::-1].iterrows():
        pnl_str = f"{row['profit_pct']:.2%}"
        report_lines.append(f"| {row['exit_date']} | {row['code']} | 卖出 | {pnl_str} | {row['duration']}天 | {row['reason']} |")

    report_path = os.path.join(report_dir, 'FINAL_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"✅ 专业报告已生成: {report_path}")
    print(f"✅ 可视化图表已保存: {chart_path}")

if __name__ == '__main__':
    generate_full_report()
