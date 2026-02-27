import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_equity_robust():
    log_file = 'backtest_detailed_log.json'
    if not os.path.exists(log_file):
        print(f"❌ Error: {log_file} not found")
        return

    # 1. 加载数据
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    daily_records = data.get('daily_records', [])
    if not daily_records:
        print("❌ Error: No daily records in log file")
        return

    df = pd.DataFrame(daily_records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 2. 计算净值与回撤
    initial_cash = df['total_value'].iloc[0]
    df['equity'] = df['total_value'] / initial_cash
    df['cum_max'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['cum_max']) / df['cum_max']

    # 3. 绘图 (使用标准配置，避开中文字体坑)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # 上图：净值
    ax1.plot(df['date'], df['equity'], color='blue', linewidth=1.5, label='Strategy Equity')
    ax1.set_title('KSP Strategy Equity Curve (2025)', fontsize=14)
    ax1.set_ylabel('Net Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 下图：回撤
    ax2.fill_between(df['date'], df['drawdown'], 0, color='red', alpha=0.3, label='Drawdown')
    ax2.set_ylabel('Drawdown %')
    ax2.set_ylim(df['drawdown'].min() * 1.2, 0.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    save_path = 'output/equity_curve_v2.png'
    plt.savefig(save_path, dpi=120)
    print(f"✅ Robust plot saved to: {save_path}")
    print(f"📈 Total points plotted: {len(df)}")
    print(f"📊 Final Equity: {df['equity'].iloc[-1]:.4f}")

if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    plot_equity_robust()
