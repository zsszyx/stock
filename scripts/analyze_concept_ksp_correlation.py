import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock.database.factory import RepositoryFactory
from stock.config import settings
from stock.data_context.concept_context_v2 import ConceptContext

def analyze_concept_ksp():
    repo = RepositoryFactory.get_clickhouse_repo()
    
    print("📥 Loading stock daily data for concept analysis...")
    cols = "date, code, close, ksp_sum_5d, ksp_sum_10d, ksp_sum_5d_rank"
    query = f"SELECT {cols} FROM {settings.TABLE_DAILY} WHERE date >= '2025-01-01' AND close > 0"
    stock_df = repo.query(query)
    
    if stock_df.empty:
        print("❌ No stock data found.")
        return

    print("🔄 Loading concept data...")
    concept_ctx = ConceptContext(repo=repo)
    
    print("📊 Calculating stock future returns...")
    stock_df = stock_df.sort_values(['code', 'date'])
    stock_df['f_ret_5d'] = stock_df.groupby('code')['close'].shift(-5) / stock_df['close'] - 1
    
    print("🔗 Mapping stocks to concepts...")
    all_concept_data = []
    for concept in concept_ctx.all_concepts:
        stocks = concept_ctx.get_stocks(concept)
        for s in stocks:
            all_concept_data.append({'concept': concept, 'code': s})
    
    concept_mapping_df = pd.DataFrame(all_concept_data)
    merged_df = stock_df.merge(concept_mapping_df, on='code')
    
    print(f"📈 Aggregating to concept level... (Rows after merge: {len(merged_df)})")
    concept_daily = merged_df.groupby(['date', 'concept']).agg({
        'ksp_sum_5d': 'mean',
        'ksp_sum_10d': 'mean',
        'f_ret_5d': 'mean'
    }).reset_index()
    
    print("🏆 Calculating concept ranks...")
    concept_daily['concept_ksp_rank'] = concept_daily.groupby('date')['ksp_sum_5d'].rank(ascending=False, method='min')
    
    analysis_df = concept_daily.dropna(subset=['f_ret_5d']).copy()
    
    print("\n" + "="*60)
    print("📊 Concept-Level KSP Predictive Power (Rank IC)")
    print("="*60)
    
    ic, p_val = spearmanr(-analysis_df['concept_ksp_rank'], analysis_df['f_ret_5d'])
    print(f"Concept KSP Rank vs Future 5D Return IC: {ic:.4f} (p-value: {p_val:.4e})")

    print("\n" + "="*60)
    print("🎯 Concept Decile Analysis (Mean Future 5D Return)")
    print("="*60)
    
    analysis_df['decile'] = analysis_df.groupby('date')['concept_ksp_rank'].transform(
        lambda x: pd.qcut(x + np.random.normal(0, 1e-6, len(x)), 10, labels=False, duplicates='drop')
    )
    
    group_rets = analysis_df.groupby('decile')['f_ret_5d'].mean()
    for i in sorted(group_rets.index):
        print(f"  Decile {i+1} (Top {(i)*10}%-{(i+1)*10}%): {group_rets[i]:.4%}")

    print("\n" + "="*60)
    print("🔝 Performance of Top 5 Concepts Daily")
    print("="*60)
    top_5_performance = analysis_df[analysis_df['concept_ksp_rank'] <= 5]['f_ret_5d'].mean()
    print(f"Daily Top 5 Concepts Mean Future 5D Return: {top_5_performance:.4%}")
    
    market_avg = analysis_df['f_ret_5d'].mean()
    print(f"Market (Concept Avg) Mean Future 5D Return: {market_avg:.4%}")

if __name__ == "__main__":
    analyze_concept_ksp()
