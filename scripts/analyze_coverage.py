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
    LiquidityFilterStep, FinalSelectionStep, SelectorContext
)

def analyze_concept_coverage():
    repo = RepositoryFactory.get_clickhouse_repo()
    latest_date_res = repo.query(f"SELECT max(date) as d FROM {settings.TABLE_DAILY}")
    latest_date_str = latest_date_res.iloc[0]['d']
    dt_obj = datetime.strptime(latest_date_str, '%Y-%m-%d')
    
    load_start = (dt_obj - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    daily_df = repo.query(f"SELECT * FROM {settings.TABLE_DAILY} WHERE date >= '{load_start}'")
    daily_ctx = DailyContext(daily_df=daily_df)
    concept_ctx = ConceptContext(repo=repo)
    
    # 手动运行漏斗以捕获 ctx
    ctx = SelectorContext(daily_ctx, concept_ctx, dt_obj)
    steps = [
        InitialUniverseStep(ksp_period=5),
        RangeConceptRankingStep(start_rank=20, end_rank=100, top_n=3, ksp_period=5),
        LiquidityFilterStep(min_amount=50000000),
        FinalSelectionStep() 
    ]
    
    for step in steps:
        ctx = step.process(ctx)
    
    candidates = ctx.pool['code'].tolist()
    selected_concepts = ctx.top_concepts
    
    day_data = daily_df[daily_df['date'] == latest_date_str]
    qualified_codes = []
    for code in candidates:
        row = day_data[day_data['code'] == code]
        if not row.empty:
            rank = row['ksp_sum_5d_rank'].iloc[0]
            if 440 <= rank <= 1300:
                qualified_codes.append(code)

    concept_counts = {}
    for code in qualified_codes:
        stock_concepts = concept_ctx.get_concept_by_stock(code)
        for cp in stock_concepts:
            if cp in selected_concepts:
                concept_counts[cp] = concept_counts.get(cp, 0) + 1
    
    sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\n📊 选股池概念覆盖度分析 (基于 {} 数据):".format(latest_date_str))
    print("入选个股总数: {}".format(len(qualified_codes)))
    print("选中的次优 Top 3 概念: {}".format(selected_concepts))
    print("-" * 60)
    for cp, count in sorted_concepts:
        coverage = count / len(qualified_codes)
        print("概念: {:<20} | 覆盖个股数: {:>2} | 覆盖率: {:.1%}".format(cp, count, coverage))

if __name__ == "__main__":
    analyze_concept_coverage()
