import os
import sys
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock.database.factory import RepositoryFactory
from stock.data_context.context import DailyContext
from stock.config import settings

def refresh_all_factors():
    repo = RepositoryFactory.get_clickhouse_repo()
    
    print("📥 Loading all data from daily_kline...")
    df = repo.query(f"SELECT * FROM {settings.TABLE_DAILY}")
    
    if df.empty:
        print("❌ No data found in daily_kline.")
        return

    print(f"📊 Loaded {len(df)} records. Processing factors and GLOBAL ranks...")
    
    # 按照代码和日期排序，确保滚动计算正确
    df = df.sort_values(['code', 'date'])
    
    # 直接在全量 DataFrame 上调用派生因子计算逻辑
    # 注意：DailyContext._add_derived_factors 内部会调用 KSPFactorEngine.add_rolling_factors
    # 而 add_rolling_factors 包含正确的 groupby('code') 和 groupby('date') 逻辑
    updated_df = DailyContext._add_derived_factors(df)

    print("🔄 Preparing for upload...")
    # 确保列顺序与数据库一致
    final_df = updated_df[DailyContext.COLUMNS]
    
    # 强制转换整数列，避免 1.0 这种格式导致 ClickHouse 无法解析 Int32/Int64
    int_cols = ['volume', 'ksp_rank', 'ksp_sum_14d_rank', 'ksp_sum_10d_rank', 'ksp_sum_7d_rank', 'ksp_sum_5d_rank', 'list_days']
    for col in int_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0).astype(int)

    print(f"📤 Inserting {len(final_df)} updated records back to {settings.TABLE_DAILY}...")
    repo.insert_df(final_df, settings.TABLE_DAILY)
    
    print("✅ All factors and global ranks refreshed successfully.")
    print("🧹 Optimizing table...")
    repo.optimize_table(settings.TABLE_DAILY)
    print("✨ Done.")

if __name__ == "__main__":
    refresh_all_factors()
