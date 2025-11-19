import pandas as pd
import numpy as np
from prepare import get_stock_merge_industry_table
from metric import atr

def calculate_dynamic_barriers(df, atr_period=14, upper_multiplier=2.0, lower_multiplier=1.5, time_barrier=5):
    """
    计算动态障碍位
    """
    # 计算ATR
    df = atr(df, timeperiod=atr_period)
    atr_col = f'atr{atr_period}'
    
    # 计算动态障碍位
    df['upper_barrier'] = df['close'] + (upper_multiplier * df[atr_col])
    df['lower_barrier'] = df['close'] - (lower_multiplier * df[atr_col])
    df['time_barrier'] = time_barrier  # 时间障碍天数
    
    return df

def create_labels_with_barriers(df, forward_days=5):
    """
    根据障碍位创建标签
    """
    # 计算未来N天内的最高价和最低价
    df['future_high'] = df['high'].rolling(window=forward_days).max().shift(-forward_days+1)
    df['future_low'] = df['low'].rolling(window=forward_days).min().shift(-forward_days+1)
    
    # 初始化标签列
    df['label'] = 0  # 0: 持平, 1: 上涨, -1: 下跌
    
    # 设置标签逻辑
    # 1: 价格上涨突破上轨
    # -1: 价格下跌突破下轨
    # 0: 在时间内未突破任一轨道
    
    # 检查是否突破上轨
    upper_breakthrough = (df['future_high'] >= df['upper_barrier'])
    # 检查是否突破下轨
    lower_breakthrough = (df['future_low'] <= df['lower_barrier'])
    
    # 设置标签
    df.loc[upper_breakthrough & ~lower_breakthrough, 'label'] = 1  # 上涨
    df.loc[lower_breakthrough & ~upper_breakthrough, 'label'] = -1  # 下跌
    # 默认为0（持平），表示在时间窗口内未突破任一轨道
    
    return df

# 加载数据
df = get_stock_merge_industry_table(length=100)

print("原始数据信息:")
print(f"总记录数: {len(df)}")
print(f"索引类型: {type(df.index)}")

# 获取所有日期
all_dates = sorted(df.index.get_level_values('date').unique())
print(f"唯一日期数量: {len(all_dates)}")
print(f"日期范围: {all_dates[0]} 到 {all_dates[-1]}")

# 计算ATR和障碍位
print("\n计算ATR和障碍位...")
df = df.groupby('code').apply(lambda x: calculate_dynamic_barriers(x))

# 创建标签
print("创建标签...")
df = df.groupby('code').apply(lambda x: create_labels_with_barriers(x))

print(f"处理后的数据记录数: {len(df)}")

# 模拟滚动窗口训练中的数据筛选
n_days = 30
m_days = 10
start_idx = n_days

print(f"\n模拟滚动窗口训练:")
print(f"n_days: {n_days}, m_days: {m_days}, start_idx: {start_idx}")
print(f"all_dates长度: {len(all_dates)}")

if start_idx + m_days <= len(all_dates):
    # 定义训练和预测时间窗口
    train_start_date = all_dates[start_idx - n_days]
    train_end_date = all_dates[start_idx - 1]  # 训练结束日期是开始预测的前一天
    predict_start_date = all_dates[start_idx]
    predict_end_date = all_dates[start_idx + m_days - 1]
    
    print(f"\n训练窗口: {train_start_date} 到 {train_end_date}")
    print(f"预测窗口: {predict_start_date} 到 {predict_end_date}")
    
    # 尝试获取训练数据
    print("\n尝试获取训练数据...")
    try:
        # 使用原始代码的方式
        train_data = df.loc[(slice(None), slice(train_start_date, train_end_date)), :]
        print(f"使用slice方式获取的训练数据记录数: {len(train_data)}")
        
        if len(train_data) > 0:
            print("训练数据前几行索引:")
            print(train_data.head().index)
            
            # 检查日期范围
            train_dates = train_data.index.get_level_values('date').unique()
            print(f"训练数据日期范围: {min(train_dates)} 到 {max(train_dates)}")
    except Exception as e:
        print(f"使用slice方式获取训练数据时出错: {e}")
    
    # 尝试另一种方式获取数据
    print("\n尝试另一种方式获取训练数据...")
    try:
        # 先获取所有股票代码
        codes = df.index.get_level_values('code').unique()
        print(f"股票代码数量: {len(codes)}")
        
        # 分别获取每个股票在时间范围内的数据
        filtered_data_list = []
        for code in codes[:5]:  # 只检查前5个股票
            stock_data = df.loc[code]
            # 筛选日期范围内的数据
            mask = (stock_data.index >= train_start_date) & (stock_data.index <= train_end_date)
            filtered_stock_data = stock_data[mask]
            if len(filtered_stock_data) > 0:
                # 重新添加股票代码层级
                filtered_stock_data['code'] = code
                filtered_data_list.append(filtered_stock_data)
                print(f"股票 {code} 在训练期间有 {len(filtered_stock_data)} 条记录")
        
        if filtered_data_list:
            combined_data = pd.concat(filtered_data_list)
            print(f"合并后的数据记录数: {len(combined_data)}")
        else:
            print("没有找到符合条件的数据")
    except Exception as e:
        print(f"使用另一种方式获取训练数据时出错: {e}")
else:
    print("索引超出范围，无法进行窗口训练")