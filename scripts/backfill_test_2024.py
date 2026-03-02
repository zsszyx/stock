import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.getcwd())

from stock.tasks.min5_update import Min5UpdateTask
from stock.tasks.daily_update import DailyAggregationTask
from scripts.refresh_factors import refresh_all_factors

def backfill_test():
    # 仅补录 2024-06 一个月的数据进行验证
    start_date = "2024-06-01"
    end_date = "2024-06-30"
    
    print("="*70)
    print(f"🚀 压力测试：补录 2024-06 数据")
    print("="*70)
    
    min5_task = Min5UpdateTask(max_workers=6, batch_size=50)
    try:
        # 使用 2024-06-03 (周一) 作为股票列表基准日，避开今日列表可能导致的 2024 请求权限问题
        min5_task.run(start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"❌ 失败: {e}")
    finally:
        min5_task.close()

if __name__ == "__main__":
    backfill_test()
