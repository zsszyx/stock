import sys
import os
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.getcwd())

from stock.tasks.min5_update import Min5UpdateTask
from stock.tasks.daily_update import DailyAggregationTask
from scripts.refresh_factors import refresh_all_factors

def full_update_pipeline():
    print("="*70)
    print("🚀 启动全量数据更新流程")
    print("="*70)
    
    target_date = "2026-02-26"
    
    # 步骤 1: 更新 5 分钟线
    print("\n[Step 1/3] 正在抓取 5 分钟线数据 ({})...".format(target_date))
    min5_task = Min5UpdateTask(max_workers=8)
    try:
        min5_task.run(start_date=target_date, end_date=target_date)
    except Exception as e:
        print("❌ 分钟线更新失败: {}".format(e))
    finally:
        min5_task.close()

    # 步骤 2: 聚合日线数据
    print("\n[Step 2/3] 正在从分钟线聚合日线数据...")
    daily_task = DailyAggregationTask(chunk_size=1)
    try:
        daily_task.run(start_date=target_date, end_date=target_date, clear_target=True)
    except Exception as e:
        print("❌ 日线聚合失败: {}".format(e))
        return
    finally:
        daily_task.close()

    # 步骤 3: 刷新全局因子与排名
    print("\n[Step 3/3] 正在计算 KSP 分数并刷新全局排名...")
    try:
        refresh_all_factors()
    except Exception as e:
        print("❌ 因子刷新失败: {}".format(e))
        return

    print("\n" + "="*70)
    print("✅ 全量数据更新已完成！")
    print("="*70)

if __name__ == "__main__":
    full_update_pipeline()
