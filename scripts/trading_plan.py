import json
from datetime import datetime
import pandas as pd
import sys
import os
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stock.database.factory import RepositoryFactory
from stock.config import settings
from stock.data_context.context import DailyContext
from stock.data_context.concept_context_v2 import ConceptContext
from stock.selector.funnel import (
    FunnelSelector, InitialUniverseStep, ConceptRankingStep, 
    LiquidityFilterStep, FinalSelectionStep
)

def generate_trading_plan(entry_rank=300, sell_rank=300, tp=0.099, sl=-0.02):
    repo = RepositoryFactory.get_clickhouse_repo()
    
    # 1. 获取最新持仓
    log_files = sorted([f for f in os.listdir('logs') if f.startswith('KSP_V7_RANK300')], reverse=True)
    if not log_files:
        print("Error: No log files found for RANK300.")
        return
    
    with open(f"logs/{log_files[0]}", 'r') as f:
        log_data = json.load(f)
    
    last_record = log_data['daily_records'][-1]
    current_date_str = last_record['date']
    positions = {p['code']: p for p in last_record['positions']}
    
    print(f"--- 📅 当前状态 (截至 {current_date_str}) ---")
    print(f"总资产: {last_record['total_value']:,.2f} | 现金: {last_record['cash']:,.2f}")
    print(f"当前持仓 ({len(positions)} 只):")
    for code, pos in positions.items():
        print(f"  - {code}: 现价 {pos['price']:.2f}, 成本 {pos['buy_price']:.2f}, 盈亏 {pos['profit_pct']:.2%}")

    # 2. 预测/选股 (针对 2026-02-26)
    target_date = datetime(2026, 2, 26)
    load_start = (target_date - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    
    daily_df = repo.query(f"SELECT * FROM {settings.TABLE_DAILY} WHERE date >= '{load_start}' AND date < '2026-02-26'")
    if daily_df.empty:
        print("Error: No daily data found.")
        return
        
    daily_ctx = DailyContext(daily_df=daily_df)
    concept_ctx = ConceptContext(repo=repo)
    computed_df = daily_ctx.data
    
    steps = [
        InitialUniverseStep(),
        ConceptRankingStep(top_n=3),
        LiquidityFilterStep(min_amount=50000000),
        FinalSelectionStep(top_n_per_concept=3)
    ]
    selector = FunnelSelector(daily_ctx, concept_ctx, steps)
    
    last_trading_date = datetime.strptime(current_date_str, '%Y-%m-%d')
    new_selection = selector.select(last_trading_date)
    
    print(f"\n--- 🎯 2026-02-26 交易计划 (TP: {tp:.0%}, SL: {sl:.0%}) ---")
    
    # A. 卖出逻辑
    sells = []
    for code, pos in positions.items():
        code_data = computed_df[(computed_df['code'] == code) & (computed_df['date'] == current_date_str)]
        rank = code_data['ksp_sum_5d_rank'].iloc[0] if not code_data.empty else 999
        
        reason = None
        if pos['profit_pct'] >= tp: reason = f"触发止盈 (>={tp:.0%})"
        elif pos['profit_pct'] <= sl: reason = f"触发止损 (<={sl:.0%})"
        elif rank > sell_rank: reason = f"排名跌破退出线 (当前排名: {int(rank)} > {sell_rank})"
        
        if reason:
            sells.append((code, reason))

    if sells:
        print("🔴 建议卖出:")
        for code, reason in sells:
            print(f"  - {code}: {reason}")
    else:
        print("🟢 无建议卖出 (持仓均符合策略要求)")

    # B. 买入逻辑
    slots = 5
    available_slots = slots - (len(positions) - len(sells))
    
    if available_slots > 0:
        buys = []
        for code in new_selection:
            if code not in positions:
                code_data = computed_df[(computed_df['code'] == code) & (computed_df['date'] == current_date_str)]
                rank = code_data['ksp_sum_5d_rank'].iloc[0] if not code_data.empty else 999
                
                if rank <= entry_rank:
                    buys.append((code, rank))
                if len(buys) >= available_slots: break
        
        if buys:
            print(f"🔵 建议买入 (可用仓位: {available_slots}):")
            for code, rank in buys:
                code_data = computed_df[(computed_df['code'] == code) & (computed_df['date'] == current_date_str)]
                poc = code_data['poc'].iloc[0] if not code_data.empty else 0.0
                print(f"  - {code}: 排名 {int(rank)}, 建议执行价(POC) {poc:.2f}")
        else:
            print(f"⚪️ 无符合买入门槛 ({entry_rank}) 的新信号")
    else:
        print("⚪️ 仓位已满，暂无买入空间")

if __name__ == "__main__":
    generate_trading_plan(entry_rank=300, sell_rank=300, tp=0.099, sl=-0.02)
