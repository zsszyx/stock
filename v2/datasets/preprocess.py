import numpy as np
import pandas as pd
import sqlite3
import random

# 特征字段列表
feature_fields = [
    'date',
    'open_ma30_ratio', 
    'high_ma30_ratio', 
    'low_ma30_ratio', 
    'close_ma30_ratio', 
    'volume_vol_ma30_ratio',
    'pctChg',
    'turn',
    'peTTM',
    'pbMRQ',
    'psTTM',
    'pcfNcfTTM'
]

class DatasetsPipeline:
    def __init__(self):
        self.plugins = []

    def register(self, plugin):
        """注册一个数据集处理插件"""
        self.plugins.append(plugin)

    def run(self, df):
        """按顺序执行所有插件"""
        for plugin in self.plugins:
            df = plugin(df)
        return df

def select_and_sort_features(df, feature_fields=feature_fields, date_field='date'):
    """
    取出特征字段并按时间升序排列
    :param df: DataFrame
    :param feature_fields: 特征字段列表
    :param date_field: 时间字段名
    :return: 新的DataFrame
    """
    df = df[feature_fields]
    df = df.sort_values(by=date_field, ascending=True).reset_index(drop=True)
    # print(df)
    return df[feature_fields[1:]]

def filter_by_max_growth(df,window=5):
    """
    检查每个5天窗口内的最大涨幅是否小于5%
    如果是，则filter=0，否则filter=1
    
    Args:
        df: 包含'pctChg'列的DataFrame，假定已按日期排序
        
    Returns:
        添加filter列的DataFrame
    """
    # 计算5天窗口内的最大涨幅
    max_pct_change = df['pctChg'].rolling(window=window).max()
    # print(max_pct_change)
    
    # 设置filter列：如果最大涨幅<5则为0，否则为1
    df['filter'] = (max_pct_change < 5).astype(int)
    
    # 处理前4行（窗口不足5天的情况）
    df.loc[df.index[:window-1], 'filter'] = 0
    
    return df

def label_by_future_pctchg_sum(df, window=5, threshold=10):
    """
    统计每一天后5天pctChg求和是否大于8，如果是label为1，否则为0
    标记window天后启动（包括当天）
    :param df: 包含'pctChg'列的DataFrame，假定已按日期排序
    :param window: 未来天数窗口
    :param threshold: 阈值
    :return: 新的DataFrame，增加'label'列
    """
    label = np.zeros(len(df))
    for i in range(len(df)-window+1):
        pctchg_sum = df['pctChg'].iloc[i:i+window].sum()
        if pctchg_sum > threshold:
            label[i] = 1
        else:
            label[i] = 0
    df['label'] = label
    # 最后window行没有未来数据，filter 为 0
    df.loc[df.index[-window+1:], 'filter'] = 0
    return df

def pecent_to_float(df):
    df['pctChg'] = df['pctChg'].astype(float) / 100.0
    df['turn'] = df['turn'].astype(float) / 100.0
    return df
# 使用示例
dataset_pipeline = DatasetsPipeline()
dataset_pipeline.register(select_and_sort_features)
dataset_pipeline.register(filter_by_max_growth)
dataset_pipeline.register(label_by_future_pctchg_sum)
dataset_pipeline.register(pecent_to_float)

if __name__ == "__main__":
    test_df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=12),
    'pctChg': [1, 2, 3, 4, 2, 3,2,1, 1, 1, 1, 9],
    'filter': [1]*12
    })

    processed_df = label_by_future_pctchg_sum(test_df)
    print(processed_df)
