import pandas as pd
from datetime import datetime
import sys
import os

sys.path.insert(0, os.getcwd())
from stock.database.factory import RepositoryFactory
from stock.data_context.context import DailyContext
from stock.data_context.concept_context_v2 import ConceptContext
from stock.selector.funnel import (
    FunnelSelector, InitialUniverseStep, ConceptRankingStep, 
    LiquidityFilterStep, FinalSelectionStep, WyckoffVolatilityStep
)

def generate_plan():
    print("🔮 计算 KSP V7 Pro 交易计划...")
    repo = RepositoryFactory.get_clickhouse_repo()
    latest_date_str = repo.query("SELECT max(date) FROM daily_kline").iloc[0,0]
    latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
    print(f"依据最新行情: {latest_date_str}")

    daily_df = repo.query(f"SELECT * FROM daily_kline WHERE date >= '2025-10-01'")
    daily_ctx = DailyContext(daily_df=daily_df)
    concept_ctx = ConceptContext(repo=repo)
    
    steps = [
        InitialUniverseStep(),
        ConceptRankingStep(top_n=3),
        LiquidityFilterStep(min_amount=50000000),
        WyckoffVolatilityStep(window=30, percentile=0.30),
        FinalSelectionStep(top_n_per_concept=3)
    ]
    selector = FunnelSelector(daily_ctx, concept_ctx, steps)
    targets = selector.select(latest_date)
    
    current_holdings = ['sh.600351', 'sh.600664', 'sh.600188', 'sh.600718', 'sh.600736', 'sh.600163', 'sh.603118', 'sh.600930']
    
    print("\n" + "="*50 + "\n🚀 交易计划\n" + "="*50)
    for code in targets:
        status = "[已持仓]" if code in current_holdings else "[新机会]"
        info = daily_df[(daily_df['code'] == code) & (daily_df['date'] == latest_date_str)].iloc[0]
        print(f"- {code} {status} | 分数:{info['ksp_score']:.2f} | POC:{info['poc']:.2f}")

    new_targets = [t for t in targets if t not in current_holdings]
    if len(current_holdings) < 9 and new_targets:
        print(f"\n👉 建议买入: {new_targets[0]} (参考POC)")
    print("="*50)
    repo.close()

if __name__ == "__main__":
    generate_plan()
