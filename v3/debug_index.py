import pandas as pd
from prepare import get_stock_merge_industry_table

# 加载数据
df = get_stock_merge_industry_table(length=100)

print("数据索引信息:")
print(f"索引类型: {type(df.index)}")
print(f"索引名称: {df.index.names}")

# 查看索引的前几项
print("\n索引前5项:")
print(df.index[:5])

# 查看索引的层级
print("\n索引层级信息:")
for i, level_name in enumerate(df.index.names):
    print(f"层级 {i}: {level_name}")
    unique_values = df.index.get_level_values(i).unique()
    print(f"  唯一值数量: {len(unique_values)}")
    print(f"  前5个唯一值: {unique_values[:5].tolist()}")

# 检查索引的排序
print("\n检查索引排序:")
# 获取第一个层级的所有值
level_0_values = df.index.get_level_values(0).unique()
print(f"第一层级前5个值: {level_0_values[:5]}")

# 获取第二个层级的所有值
level_1_values = df.index.get_level_values(1).unique()
print(f"第二层级前5个值: {level_1_values[:5]}")

# 尝试不同的索引访问方式
print("\n尝试不同的索引访问方式:")
try:
    # 方式1: 使用层级名称
    first_code = df.index.get_level_values('code')[0]
    first_date = df.index.get_level_values('date')[0]
    print(f"使用层级名称访问: code={first_code}, date={first_date}")
    
    # 方式2: 使用位置索引
    first_index = df.index[0]
    print(f"使用位置索引访问: {first_index}")
    
    # 方式3: 尝试获取特定股票和日期的数据
    sample_code = df.index.get_level_values('code').unique()[0]
    sample_date = df.index.get_level_values('date').unique()[30]  # 取中间的一个日期
    print(f"\n尝试获取股票 {sample_code} 在日期 {sample_date} 的数据:")
    
    # 使用xs方法
    subset = df.xs(sample_date, level='date')
    print(f"使用xs方法获取日期 {sample_date} 的数据记录数: {len(subset)}")
    
    # 使用xs方法获取特定股票和日期
    try:
        single_record = df.xs((sample_code, sample_date), level=('code', 'date'))
        print(f"使用xs方法获取股票 {sample_code} 在日期 {sample_date} 的数据记录数: {len(single_record)}")
    except:
        print("无法使用xs方法获取特定股票和日期的数据")
        
except Exception as e:
    print(f"访问索引时出错: {e}")

# 检查如何正确地筛选日期范围内的数据
print("\n检查如何正确筛选日期范围内的数据:")
all_dates = sorted(df.index.get_level_values('date').unique())
train_start_date = all_dates[0]
train_end_date = all_dates[29]  # 前30天

print(f"训练日期范围: {train_start_date} 到 {train_end_date}")

# 方法1: 使用query方法
try:
    filtered_df = df.query('date >= @train_start_date and date <= @train_end_date')
    print(f"使用query方法筛选的数据记录数: {len(filtered_df)}")
except Exception as e:
    print(f"使用query方法时出错: {e}")

# 方法2: 使用xs方法筛选日期范围
try:
    # 先获取日期范围内的所有数据
    date_mask = (df.index.get_level_values('date') >= train_start_date) & (df.index.get_level_values('date') <= train_end_date)
    filtered_df = df[date_mask]
    print(f"使用布尔索引筛选的数据记录数: {len(filtered_df)}")
except Exception as e:
    print(f"使用布尔索引时出错: {e}")