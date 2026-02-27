import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# 设置样式
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # macOS 字体
plt.rcParams['axes.unicode_minus'] = False

def plot_equity_curve():
    json_file = 'backtest_detailed_log.json'
    if not os.path.exists(json_file):
        print(f"❌ 找不到数据文件: {json_file}")
        return

    # 1. 加载策略数据
    import json
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    daily_records = data.get('daily_records', [])
    if not daily_records:
        print("❌ 日志文件中没有每日记录")
        return
        
    df = pd.DataFrame(daily_records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 归一化策略净值
    initial_value = 1000000.0 # 初始资金
    df['strategy_equity'] = df['total_value'] / initial_value

    # 2. 模拟基准数据 (由于此处无法直接联网获取实时指数，我们从每日涨跌幅估算或展示策略净值)
    # 计算回撤
    df['cum_max'] = df['strategy_equity'].cummax()
    df['drawdown'] = (df['strategy_equity'] - df['cum_max']) / df['cum_max']

    # 3. 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # 主图：收益曲线
    ax1.plot(df['date'], df['strategy_equity'], label='KSP 策略 (300排名版)', color='#1f77b4', linewidth=2)
    ax1.set_title('KSP 策略回测收益曲线 (2025-01-01 至今)', fontsize=16)
    ax1.set_ylabel('累计净值', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 副图：回撤
    ax2.fill_between(df['date'], df['drawdown'], 0, color='red', alpha=0.3, label='最大回撤')
    ax2.set_ylabel('回撤 %', fontsize=12)
    ax2.set_ylim(-0.20, 0.02)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    # 格式化 X 轴
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=0)

    # 保存图片
    os.makedirs('output', exist_ok=True)
    save_path = 'output/equity_curve.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ 收益曲线已保存至: {save_path}")

if __name__ == '__main__':
    plot_equity_curve()
