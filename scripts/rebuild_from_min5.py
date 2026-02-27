import sys
import os
from datetime import datetime
sys.path.insert(0, os.getcwd())

from stock.tasks.daily_update import DailyAggregationTask

def rebuild():
    """
    使用 ClickHouse 中已有的 5 分钟数据重新构建日线数据及因子
    """
    print("="*70)
    print("🛠️  独立任务：从 5 分钟线重构日线表 (Decoupled Mode)")
    print("="*70)
    
    # 获取参数，默认恢复 2025 年至今
    start_date = "2025-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    task = DailyAggregationTask(chunk_size=3) # 设置较小的 chunk 以降低内存压力
    try:
        # clear_target=True 确保重构时不会产生重复数据
        task.run(start_date=start_date, end_date=end_date, clear_target=True)
    finally:
        task.close()

if __name__ == "__main__":
    rebuild()
