import pandas as pd
from prepare import get_stock_merge_industry_table

# 加载数据
df = get_stock_merge_industry_table(length=100)

print("数据基本信息:")
print(f"总记录数: {len(df)}")
print(f"股票数量: {df.index.get_level_values('code').nunique()}")
print(f"日期范围: {df.index.get_level_values('date').min()} 到 {df.index.get_level_values('date').max()}")

# 查看每个股票的数据量
stock_counts = df.groupby('code').size()
print(f"\n每个股票的记录数:")
print(stock_counts.describe())

# 查看前几条记录
print("\n前5条记录:")
print(df.head())

# 查看索引信息
print("\n索引信息:")
print(df.index)