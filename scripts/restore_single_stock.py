import pandas as pd
import baostock as bs
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.getcwd())
from stock.tasks.min5_update import Min5UpdateTask, fetch_data_task, init_worker
from stock.tasks.daily_update import DailyUpdateTask
from stock.database.factory import RepositoryFactory
from stock.config import settings
from stock.data_fetch.data_provider.baostock_provider import BaoInterface

def restore_single_stock(code="sh.600000"):
    print(f"🚀 Starting test restoration for {code} (2025-01-01 to now)")
    
    repo = RepositoryFactory.get_clickhouse_repo()
    repo.create_mintues5_table()
    repo.create_daily_kline_table()
    
    start_date = "2025-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 1. Fetch 5-minute data
    print(f"⏳ Fetching 5-minute data for {code}...")
    with BaoInterface() as bi:
        df_min5 = bi.get_k_data_5min(code=code, start_date=start_date, end_date=end_date)
    
    if df_min5 is not None and not df_min5.empty:
        print(f"✅ Fetched {len(df_min5)} rows of 5-minute data.")
        df_min5['volume'] = df_min5['volume'].astype(int)
        repo.insert_df(df_min5, settings.TABLE_MIN5)
    else:
        print(f"❌ Failed to fetch data for {code} or data is empty.")
        return

    # 2. Aggregate to daily
    print(f"⏳ Aggregating to daily data...")
    daily_task = DailyUpdateTask()
    try:
        # We can't easily run it for just one stock with the current DailyUpdateTask.run 
        # because it processes by dates. But since we only have one stock's data in min5 table (assuming it was empty),
        # it should work fine.
        daily_task.run(full_recompute=True)
    except Exception as e:
        print(f"❌ Daily aggregation failed: {e}")
    finally:
        daily_task.close()
    
    # 3. Verify
    print("🔍 Verifying data in ClickHouse...")
    res_min5 = repo.query(f"SELECT count() as count FROM {settings.TABLE_MIN5} WHERE code = '{code}'")
    res_daily = repo.query(f"SELECT count() as count FROM {settings.TABLE_DAILY} WHERE code = '{code}'")
    
    print(f"📊 Min5 rows for {code}: {res_min5['count'][0]}")
    print(f"📊 Daily rows for {code}: {res_daily['count'][0]}")
    
    repo.close()
    print("🏁 Test restoration complete.")

if __name__ == "__main__":
    restore_single_stock()
