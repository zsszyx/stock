import pandas as pd
import baostock as bs
from datetime import datetime
import sys
import os
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.getcwd())
from stock.database.factory import RepositoryFactory
from stock.data_fetch.data_provider.baostock_provider import BaoInterface

def full_market_restore():
    print("🚀 正在启动全市场数据恢复 (2025-01-01 至今)...")
    bs.login()
    repo = RepositoryFactory.get_sqlite_repo()
    
    # 1. 获取全量股票列表 (仅限 sh.60, sz.00, sz.30)
    rs = bs.query_all_stock(day='2025-02-21')
    stock_list = []
    while (rs.error_code == '0') & rs.next():
        code = rs.get_row_data()[0]
        if code.startswith(('sh.60', 'sz.00', 'sz.30')):
            stock_list.append(code)
    
    print(f"📊 目标股票总数: {len(stock_list)}")
    
    def sync_stock(code):
        with BaoInterface() as bi:
            df = bi.get_k_data_daily(code, '2025-01-01', datetime.now().strftime('%Y-%m-%d'))
            if df is not None and not df.empty:
                for col in ['open', 'high', 'low', 'close', 'amount', 'turn']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
                
                # 补齐 KSP 必需列 (设为基础分)
                df['poc'] = df['close']
                df['ksp_score'] = 1.0
                df['ksp_sum_14d_rank'] = 100
                df['list_days'] = 500
                for col in ['ksp_rank', 'ksp_sum_7d_rank', 'ksp_sum_5d_rank', 'ksp_sum_5d']:
                    df[col] = 100
                
                return df
            return None

    # 并行抓取前 100 只用于初步恢复 (可根据需要扩大范围)
    print("⏳ 正在抓取数据 (Batch 1)...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(sync_stock, stock_list[:100]))
    
    valid_dfs = [r for r in results if r is not None]
    if valid_dfs:
        final_df = pd.concat(valid_dfs)
        repo.insert_df(final_df, "daily_kline", if_exists='append')
        print(f"✅ 成功恢复 {len(valid_dfs)} 只股票, 共 {len(final_df)} 行记录.")

    bs.logout()
    repo.close()

if __name__ == "__main__":
    full_market_restore()
