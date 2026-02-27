import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock.database.factory import RepositoryFactory
from stock.config import settings

def analyze_ksp_comprehensive():
    repo = RepositoryFactory.get_clickhouse_repo()
    
    print("📥 Loading data for comprehensive KSP analysis...")
    cols = "date, code, close, ksp_rank, ksp_sum_5d_rank, ksp_sum_7d_rank, ksp_sum_10d_rank, ksp_sum_14d_rank"
    query = f"SELECT {cols} FROM {settings.TABLE_DAILY} WHERE date >= '2025-01-01' AND close > 0 ORDER BY code, date"
    df = repo.query(query)
    
    if df.empty:
        print("❌ No data found.")
        return

    print(f"📊 Loaded {len(df)} records. Calculating metrics...")
    df['f_ret_5d'] = df.groupby('code')['close'].shift(-5) / df['close'] - 1
    
    # Rank Velocity: Change in rank over 1 day (yesterday rank - today rank)
    # A positive change means the rank improved (number decreased)
    df['rank_delta_5d'] = df.groupby('code')['ksp_sum_5d_rank'].shift(1) - df['ksp_sum_5d_rank']
    
    rank_cols = ['ksp_rank', 'ksp_sum_5d_rank', 'ksp_sum_7d_rank', 'ksp_sum_10d_rank', 'ksp_sum_14d_rank']
    
    analysis_df = df.dropna(subset=['f_ret_5d']).copy()
    
    print("\n" + "="*60)
    print("🔄 Rank Stability (Correlation between different periods)")
    print("="*60)
    stability_matrix = analysis_df[rank_cols].corr(method='spearman')
    print(stability_matrix.round(4))

    print("\n" + "="*60)
    print("📈 KSP Predictive Power Analysis (IC: Rank Correlation)")
    print("="*60)
    print(f"{'Rank Cycle':<20} | {'Spearman IC':<15}")
    print("-" * 40)
    
    for col in rank_cols:
        ic, _ = spearmanr(-analysis_df[col], analysis_df['f_ret_5d'])
        print(f"{col:<20} | {ic:>15.4f}")

    # IC for Rank Delta
    analysis_df_delta = analysis_df.dropna(subset=['rank_delta_5d'])
    ic_delta, _ = spearmanr(analysis_df_delta['rank_delta_5d'], analysis_df_delta['f_ret_5d'])
    print(f"{'rank_delta_5d':<20} | {ic_delta:>15.4f} (Higher delta = Rank Improved)")

    print("\n" + "="*60)
    print("🎯 Decile Analysis (Mean Future 5D Return by Rank Group)")
    print("="*60)
    
    decile_summary = pd.DataFrame()
    for col in rank_cols:
        noise = np.random.normal(0, 1e-6, len(analysis_df))
        analysis_df['temp_q'] = pd.qcut(analysis_df[col] + noise, 10, labels=False)
        group_rets = analysis_df.groupby('temp_q')['f_ret_5d'].mean()
        decile_summary[col] = group_rets

    decile_summary.index = [f"D{i+1}" for i in decile_summary.index]
    print(decile_summary.map(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A"))

    print("\n💡 Note: D1 represents the top 10% (best KSP rank), D10 is the bottom 10%.")
    
if __name__ == "__main__":
    analyze_ksp_comprehensive()
