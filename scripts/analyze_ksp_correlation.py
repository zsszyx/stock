import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock.database.factory import RepositoryFactory
from stock.config import settings

def analyze_ksp_stats():
    repo = RepositoryFactory.get_clickhouse_repo()
    
    print("📥 Loading data for 2025 analysis...")
    query = f"SELECT date, code, close, ksp_sum_5d_rank, ksp_sum_10d_rank FROM {settings.TABLE_DAILY} WHERE date >= '2025-01-01' AND date <= '2025-12-31' AND close > 0 ORDER BY code, date"
    df = repo.query(query)
    
    if df.empty:
        print("❌ No data found.")
        return

    print("🔄 Calculating future returns...")
    df['f_ret_1d'] = df.groupby('code')['close'].shift(-1) / df['close'] - 1
    df['f_ret_5d'] = df.groupby('code')['close'].shift(-5) / df['close'] - 1

    analysis_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['f_ret_1d', 'f_ret_5d', 'ksp_sum_5d_rank', 'ksp_sum_10d_rank']).copy()

    stability_corr, _ = spearmanr(analysis_df['ksp_sum_5d_rank'], analysis_df['ksp_sum_10d_rank'])
    
    ic_1d_5r, _ = spearmanr(analysis_df['ksp_sum_5d_rank'], analysis_df['f_ret_1d'])
    ic_5d_5r, _ = spearmanr(analysis_df['ksp_sum_5d_rank'], analysis_df['f_ret_5d'])
    ic_1d_10r, _ = spearmanr(analysis_df['ksp_sum_10d_rank'], analysis_df['f_ret_1d'])
    ic_5d_10r, _ = spearmanr(analysis_df['ksp_sum_10d_rank'], analysis_df['f_ret_5d'])

    print("\n" + "="*50)
    print("📊 KSP Stability and Correlation Report (2025)")
    print("="*50)
    print(f"Rank Stability (5D vs 10D): {stability_corr:.4f}")
    
    print("\nPredictive Power (Rank IC):")
    print(f"  - 5D Rank vs Future 1D Return:  {ic_1d_5r:.4f}")
    print(f"  - 5D Rank vs Future 5D Return:  {ic_5d_5r:.4f}")
    print(f"  - 10D Rank vs Future 1D Return: {ic_1d_10r:.4f}")
    print(f"  - 10D Rank vs Future 5D Return: {ic_5d_10r:.4f}")

    print("\nDecile Analysis (Mean Future 5D Return):")
    # Group 0 is Top 10% (Best KSP Rank)
    analysis_df['rank_group'] = pd.qcut(analysis_df['ksp_sum_5d_rank'] + np.random.normal(0, 0.01, len(analysis_df)), 10, labels=False)
    group_returns = analysis_df.groupby('rank_group')['f_ret_5d'].mean()
    
    for i in sorted(group_returns.index):
        ret = group_returns[i]
        print(f"  - Group {i+1} (Top {i*10}%-{(i+1)*10}%): {ret:.4%}")

if __name__ == "__main__":
    analyze_ksp_stats()
