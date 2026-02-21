import pandas as pd
import numpy as np
import io

def analyze_trades():
    file = 'backtest_trades.csv'
    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return
    
    # 过滤卖出记录
    sells = df[df['action'] == 'SELL_SIGNAL'].copy()
    if sells.empty:
        print("❌ 没有找到卖出信号记录。")
        return

    print("="*50)
    print("🔍 逐笔成交异常值统计分析")
    print("="*50)

    # 1. 盈亏分布异常
    print(f"总成交笔数: {len(sells)}")
    print(f"平均收益率: {sells['profit_pct'].mean():.2%}")
    
    best_idx = sells['profit_pct'].idxmax()
    worst_idx = sells['profit_pct'].idxmin()
    print(f"最大盈利: {sells.loc[best_idx, 'profit_pct']:.2%} (代码: {sells.loc[best_idx, 'code']}, 原因: {sells.loc[best_idx, 'reason']})")
    print(f"最大亏损: {sells.loc[worst_idx, 'profit_pct']:.2%} (代码: {sells.loc[worst_idx, 'code']}, 原因: {sells.loc[worst_idx, 'reason']})")

    # 2. 统计穿透止损/止盈的情况
    extreme_losses = sells[sells['profit_pct'] < -0.05]
    print(f"\n⚠️ 严重止损单 (亏损 > 5%): {len(extreme_losses)} 笔")
    if not extreme_losses.empty:
        print(extreme_losses[['date', 'code', 'profit_pct', 'reason']].sort_values('profit_pct').head(10))

    extreme_wins = sells[sells['profit_pct'] > 0.12]
    print(f"\n🚀 意外高收益单 (盈利 > 12%): {len(extreme_wins)} 笔")
    if not extreme_wins.empty:
        print(extreme_wins[['date', 'code', 'profit_pct', 'reason']].sort_values('profit_pct', ascending=False).head(10))

    # 3. 排名退出的频率
    rank_exits = sells[sells['reason'].str.contains('rank', na=False, case=False)]
    print(f"\n📊 因排名下降而卖出的笔数: {len(rank_exits)} ({len(rank_exits)/len(sells):.1%})")

    # 4. 检查是否有 0 价格或无效数值的异常
    invalid_trades = df[(df['price'] <= 0) | (df['price'].isna())]
    if not invalid_trades.empty:
        print(f"\n🚨 发现无效交易数据 (价格异常): {len(invalid_trades)} 笔")
        print(invalid_trades[['date', 'code', 'action', 'price']].head(10))
    else:
        print("\n✅ 未发现价格为负或缺失的无效交易。")

if __name__ == '__main__':
    analyze_trades()
