import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.getcwd())

from stock.tasks.normal_daily_update import NormalDailyUpdateTask
from stock.database.factory import RepositoryFactory
from stock.config import settings

def run_update():
    repo = RepositoryFactory.get_clickhouse_repo()
    
    # 获取现有日线表的日期范围作为参考
    print("🔍 正在检查现有日线表数据范围...")
    date_res = repo.query(f"SELECT min(date) as min_d, max(date) as max_d FROM {settings.TABLE_DAILY}")
    
    if not date_res.empty and date_res.iloc[0]['min_d']:
        start_date = str(date_res.iloc[0]['min_d'])
        end_date = str(date_res.iloc[0]['max_d'])
    else:
        start_date = "2025-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"📅 目标同步范围: {start_date} 至 {end_date}")
    
    task = NormalDailyUpdateTask(max_workers=10)
    try:
        task.run(start_date=start_date, end_date=end_date)
    finally:
        task.close()

if __name__ == "__main__":
    run_update()
