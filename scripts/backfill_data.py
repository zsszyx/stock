import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.getcwd())

from stock.tasks.min5_update import Min5UpdateTask
from stock.tasks.daily_update import DailyAggregationTask
from scripts.refresh_factors import refresh_all_factors

def backfill_pipeline():
    start_date = "2024-06-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print("="*70)
    print("🚀 启动历史数据大补录: {} -> {}".format(start_date, end_date))
    print("="*70)
    
    # 步骤 1: 补录 5 分钟线
    print("\n[Step 1/3] 正在抓取 5 分钟线数据...")
    min5_task = Min5UpdateTask(max_workers=10, batch_size=100)
    try:
        min5_task.run(start_date=start_date, end_date=end_date)
    except Exception as e:
        print("❌ 分钟线更新失败: {}".format(e))
    finally:
        min5_task.close()

    # 步骤 2: 重建日线数据
    print("\n[Step 2/3] 正在重构日线聚合表...")
    daily_task = DailyAggregationTask(chunk_size=5)
    try:
        daily_task.run(start_date=start_date, end_date=end_date, clear_target=True)
    except Exception as e:
        print("❌ 日线聚合失败: {}".format(e))
        return
    finally:
        daily_task.close()

    # 步骤 3: 刷新全局因子与排名
    print("\n[Step 3/3] 正在进行全量因子刷新与截面排名计算...")
    try:
        refresh_all_factors()
    except Exception as e:
        print("❌ 因子刷新失败: {}".format(e))
        return

    print("\n" + "="*70)
    print("✅ 数据补录与因子刷新圆满完成！")
    print("="*70)

if __name__ == "__main__":
    backfill_pipeline()
