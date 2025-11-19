import pandas as pd
from prepare import get_stock_merge_industry_table

# 加载数据
df = get_stock_merge_industry_table(length=100)

print("数据基本信息:")
print(f"总记录数: {len(df)}")
print(f"股票数量: {df.index.get_level_values('code').nunique()}")
print(f"日期范围: {df.index.get_level_values('date').min()} 到 {df.index.get_level_values('date').max()}")

# 获取所有日期
all_dates = sorted(df.index.get_level_values('date').unique())
print(f"唯一日期数量: {len(all_dates)}")

# 打印前几个日期
print("前10个日期:")
for i, date in enumerate(all_dates[:10]):
    print(f"  {i}: {date}")

# 打印后几个日期
print("后10个日期:")
for i, date in enumerate(all_dates[-10:]):
    print(f"  {len(all_dates)-10+i}: {date}")

# 检查特定时间窗口的数据
train_start_date = all_dates[0]
train_end_date = all_dates[30]
predict_start_date = all_dates[30]
predict_end_date = all_dates[40]

print(f"\n检查时间窗口:")
print(f"训练窗口: {train_start_date} 到 {train_end_date}")
print(f"预测窗口: {predict_start_date} 到 {predict_end_date}")

# 获取训练数据
train_data = df.loc[(slice(None), slice(train_start_date, train_end_date)), :]
print(f"训练数据记录数: {len(train_data)}")

# 获取预测数据
predict_data = df.loc[(slice(None), slice(predict_start_date, predict_end_date)), :]
print(f"预测数据记录数: {len(predict_data)}")