import os
import sys
import pandas as pd
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock.database.factory import RepositoryFactory
from stock.config import settings
from stock.data_context.context import DailyContext
from stock.data_context.concept_context_v2 import ConceptContext
from stock.selector.funnel import (
    FunnelSelector, InitialUniverseStep, RangeConceptRankingStep, 
    LiquidityFilterStep, FinalSelectionStep
)

def debug_jan():
    repo = RepositoryFactory.get_clickhouse_repo()
    test_date = "2025-01-02"
    
    # 加载数据
    load_start = "2024-11-01"
    daily_df = repo.query(f"SELECT * FROM {settings.TABLE_DAILY} WHERE date >= '{load_start}' AND date <= '2025-01-10'")
    daily_ctx = DailyContext(daily_df=daily_df)
    concept_ctx = ConceptContext(repo=repo)
    
    # 模拟漏斗
    steps = [
        InitialUniverseStep(ksp_period=5),
        RangeConceptRankingStep(start_rank=20, end_rank=100, top_n=3, ksp_period=5),
        LiquidityFilterStep(min_amount=50000000),
        FinalSelectionStep() 
    ]
    selector = FunnelSelector(daily_ctx, concept_ctx, steps)
    
    print(f"--- Debugging Funnel for {test_date} ---")
    dt_obj = datetime.strptime(test_date, '%Y-%m-%d')
    candidates = selector.select(dt_obj)
    print(f"Candidates found: {len(candidates)}")
    
    if candidates:
        day_data = daily_df[daily_df['date'] == test_date]
        qualified = []
        for code in candidates:
            row = day_data[day_data['code'] == code]
            rank = row['ksp_sum_5d_rank'].iloc[0]
            if 440 <= rank <= 1300:
                qualified.append(code)
        print(f"Qualified stocks (Rank 440-1300): {len(qualified)}")
        if qualified:
            print(f"Sample qualified codes: {qualified[:5]}")

if __name__ == "__main__":
    debug_jan()
