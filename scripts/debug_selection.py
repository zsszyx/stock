from stock.database.factory import RepositoryFactory
from stock.config import settings
from stock.data_context.context import DailyContext
from stock.data_context.concept_context_v2 import ConceptContext
from stock.selector.funnel import (
    FunnelSelector, InitialUniverseStep, ConceptRankingStep, 
    LiquidityFilterStep, FinalSelectionStep
)
from datetime import datetime, timedelta
import pandas as pd

def debug_selection():
    repo = RepositoryFactory.get_clickhouse_repo()
    start_date = "2025-01-02"
    load_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=120)).strftime('%Y-%m-%d')
    
    print(f"Loading data from {load_start_date} to 2025-02-01...")
    daily_df = repo.query(f"SELECT * FROM {settings.TABLE_DAILY} WHERE date >= '{load_start_date}' AND date <= '2025-02-01'")
    
    print(f"Total rows loaded: {len(daily_df)}")
    
    daily_ctx = DailyContext(daily_df=daily_df)
    concept_ctx = ConceptContext(repo=repo)
    
    steps = [
        InitialUniverseStep(),
        ConceptRankingStep(top_n=3),
        LiquidityFilterStep(min_amount=50000000),
        FinalSelectionStep(top_n_per_concept=3)
    ]
    selector = FunnelSelector(daily_ctx, concept_ctx, steps)
    
    test_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Manually run steps for debugging
    from stock.selector.funnel import SelectorContext
    ctx = SelectorContext(daily_ctx, concept_ctx, test_date)
    
    for step in steps:
        ctx = step.process(ctx)
        print(f"After {step.name}: Pool size = {len(ctx.pool)}")
        if not ctx.pool.empty and step.name == "Final Selection":
            print(f"Final codes: {ctx.pool['code'].tolist()}")
            # Check ranks in daily_ctx for these codes
            daily_data = daily_ctx.get_window(test_date, window_days=1, codes=ctx.pool['code'].tolist())
            print("Ranks for final codes:")
            print(daily_data[['code', 'ksp_sum_5d_rank']])

if __name__ == "__main__":
    debug_selection()
