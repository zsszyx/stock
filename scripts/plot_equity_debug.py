import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_equity_debug():
    log_file = 'backtest_detailed_log.json'
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data.get('daily_records', []))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    initial_cash = df['total_value'].iloc[0]
    df['equity'] = df['total_value'] / initial_cash

    # 创建三个子图：净值、持仓数、回撤
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)

    # 1. 净值
    ax1.plot(df['date'], df['equity'], color='blue', label='Strategy Equity')
    ax1.set_title('KSP Strategy Debug Plot', fontsize=14)
    ax1.set_ylabel('Net Value')
    ax1.grid(True, alpha=0.3)

    # 2. 持仓数 (关键诊断指标)
    ax2.step(df['date'], df['position_count'], where='post', color='green', label='Active Positions')
    ax2.set_ylabel('Position Count')
    ax2.set_ylim(0, df['position_count'].max() + 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. 回撤
    df['cum_max'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['cum_max']) / df['cum_max']
    ax3.fill_between(df['date'], df['drawdown'], 0, color='red', alpha=0.3, label='Drawdown')
    ax3.set_ylabel('Drawdown %')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/equity_debug.png')
    print("✅ Debug plot saved to: output/equity_debug.png")

if __name__ == '__main__':
    plot_equity_debug()
