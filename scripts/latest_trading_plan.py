import os
import sys
import pandas as pd
import numpy as np
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

def get_latest_plan():
    repo = RepositoryFactory.get_clickhouse_repo()
    
    # 1. 获取最新数据日期
    latest_date_res = repo.query(f"SELECT max(date) as d FROM {settings.TABLE_DAILY}")
    latest_date_str = latest_date_res.iloc[0]['d']
    print(f"🔍 数据库最新成交日: {latest_date_str}")
    
    # 3. 运行最新的选股漏斗
    load_start = (datetime.strptime(latest_date_str, '%Y-%m-%d') - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    daily_df = repo.query(f"SELECT * FROM {settings.TABLE_DAILY} WHERE date >= '{load_start}'")
    
    daily_ctx = DailyContext(daily_df=daily_df)
    concept_ctx = ConceptContext(repo=repo)
    
    steps = [
        InitialUniverseStep(ksp_period=5),
        RangeConceptRankingStep(start_rank=20, end_rank=100, top_n=3, ksp_period=5),
        LiquidityFilterStep(min_amount=50000000),
        FinalSelectionStep() 
    ]
    selector = FunnelSelector(daily_ctx, concept_ctx, steps)
    
    dt_obj = datetime.strptime(latest_date_str, '%Y-%m-%d')
    candidates = selector.select(dt_obj)
    
    day_data = daily_df[daily_df['date'] == latest_date_str]
    
    qualified_list = []
    for code in candidates:
        row = day_data[day_data['code'] == code]
        if not row.empty:
            rank = row['ksp_sum_5d_rank'].iloc[0]
            if 440 <= rank <= 1300:
                qualified_list.append({
                    'code': code,
                    'rank': rank,
                    'poc': row['poc'].iloc[0],
                    'close': row['close'].iloc[0]
                })
    
    qualified_list.sort(key=lambda x: x['rank'])
    
    print("\n" + "="*60)
    print(f"🚀 2026-02-26 交易计划 (基于 {latest_date_str} 截面数据)")
    print("="*60)
    print(f"🎯 核心逻辑: 次优概念 (Rank 20-100) + 次优个股 (Rank 440-1300)")
    print(f"📊 满足条件的个股总数: {len(qualified_list)}")
    
    print("\n💎 建议买入清单 (优先级 Top 10):")
    for i, item in enumerate(qualified_list[:10]):
        print(f"  {i+1}. {item['code']:<10} | KSP排名: {int(item['rank']):<4} | 建议买入价(POC): {item['poc']:.2f} | 前收盘: {item['close']:.2f}")

    print("\n🛑 退出风控提醒:")
    print(f"  - 排名劣化: 若持仓股 5d KSP 排名 > 1500，建议择机退出。")
    print(f"  - 尾部风险: 若持仓股 5d KSP 排名 > 3500 (D9-D10)，建议立即清仓。")
    print("="*60)

if __name__ == "__main__":
    get_latest_plan()
