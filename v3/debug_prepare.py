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

def create_features_with_lags(df, feature_columns, lag_days=5):
    """
    创建滞后特征
    """
    # 为每个特征创建滞后版本
    for col in feature_columns:
        for lag in range(lag_days):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df

def prepare_training_data(df, feature_columns, label_column='label', lag_days=5):
    """
    准备训练数据
    """
    # 创建滞后特征
    df = create_features_with_lags(df, feature_columns, lag_days)
    
    # 删除包含NaN的行（由于滞后特征产生）
    df = df.dropna()
    
    # 构建特征列名
    feature_cols = []
    for col in feature_columns:
        for lag in range(lag_days):
            feature_cols.append(f'{col}_lag_{lag}')
    
    # 提取特征和标签
    X = df[feature_cols]
    y = df[label_column]
    
    return X, y, df

# 加载数据
df = get_stock_merge_industry_table(length=100)

print("原始数据信息:")
print(f"总记录数: {len(df)}")
print(f"索引类型: {type(df.index)}")

# 获取所有日期
all_dates = sorted(df.index.get_level_values('date').unique())
print(f"唯一日期数量: {len(all_dates)}")

# 选择一个股票进行测试
test_code = df.index.get_level_values('code').unique()[0]
print(f"\n测试股票: {test_code}")

# 获取该股票的数据
stock_data = df.loc[test_code]
print(f"股票数据记录数: {len(stock_data)}")
print("股票数据前5行:")
print(stock_data.head())

# 计算ATR和障碍位
print("\n计算ATR和障碍位...")
stock_data = calculate_dynamic_barriers(stock_data)
print("计算后数据前5行:")
print(stock_data.head())

# 创建标签
print("\n创建标签...")
stock_data = create_labels_with_barriers(stock_data)
print("带标签的数据前5行:")
print(stock_data.head())

# 定义特征列
feature_columns = ['open', 'high', 'low', 'close', 'volume', 'turn', 'amount', 'pctChg', 'sh_close']

# 准备训练数据
print("\n准备训练数据...")
X, y, processed_data = prepare_training_data(stock_data, feature_columns, lag_days=3)
print(f"特征矩阵形状: {X.shape}")
print(f"标签向量形状: {y.shape}")
print("处理后的数据前5行:")
print(processed_data.head())

print("\n特征列名:")
print(X.columns.tolist())

print("\n标签值分布:")
print(y.value_counts())