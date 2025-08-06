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
    'pctChg'
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
        print(pctchg_sum)
        if pctchg_sum > threshold:
            label[i] = 1
        else:
            label[i] = 0
    df['label'] = label
    # 最后window行没有未来数据，filter 为 0
    df.loc[df.index[-window+1:], 'filter'] = 0
    return df

class SingleTableDatasetIterator:
    def __init__(self, df, window_size=60, label_field='label', filter_field='filter'):
        """
        :param df: 已处理好的DataFrame
        :param window_size: 滑动窗口长度
        :param feature_fields: 特征字段列表（不含label/filter），默认自动推断
        :param label_field: 标签字段名
        :param filter_field: 过滤字段名
        """
        self.window_size = window_size
        self.label_field = label_field
        self.filter_field = filter_field
        self.df = df
        self.shuffle = True  # 是否打乱数据顺序


    def __iter__(self):
        valid_indices = [
            idx for idx in range(self.window_size - 1, len(self.df))
            if self.df.at[idx, self.filter_field] == 1
        ]
        if self.shuffle:
            random.shuffle(valid_indices)
        for idx in valid_indices:
            window = self.df.iloc[idx - self.window_size + 1: idx + 1]
            X = window[feature_fields].values
            y = self.df.at[idx, self.label_field]
            yield X, y

class MultiTableDatasetIterator:
    def __init__(self, db_path, table_names, window_size=30, label_field='label', filter_field='filter'):
        """
        :param db_path: 数据库文件路径
        :param table_names: 需要读取的表名列表
        :param window_size: 滑动窗口长度
        :param label_field: 标签字段名
        :param filter_field: 过滤字段名
        """
        self.db_path = db_path
        self.table_names = table_names
        self.window_size = window_size
        self.label_field = label_field
        self.filter_field = filter_field
        self.shuffle = True  # 是否打乱表顺序

    def __iter__(self):
        conn = sqlite3.connect(self.db_path)
        if self.shuffle:
            random.shuffle(self.table_names)
        for table in self.table_names:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            df = dataset_pipeline.run(df)  # 使用数据集处理管道处理DataFrame
            iterator = SingleTableDatasetIterator(df, window_size=self.window_size, label_field=self.label_field, filter_field=self.filter_field)
            for X, y in iterator:
                yield X, y


# 使用示例
dataset_pipeline = DatasetsPipeline()
dataset_pipeline.register(select_and_sort_features)
dataset_pipeline.register(filter_by_max_growth)
dataset_pipeline.register(label_by_future_pctchg_sum)

test_df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=12),
    'pctChg': [1, 2, 3, 4, 2, 3,2,1, 1, 1, 1, 9],
    'filter': [1]*12
})
# 运行数据集处理管道
test_df = filter_by_max_growth(test_df)
processed_df = label_by_future_pctchg_sum(test_df)
# 输出处理后的DataFrame
print(processed_df)

# 用法示例
# for X, y in SingleTableDatasetIterator(processed_df, window_size=30):
#     print(X.shape, y)

# 用法示例
# db_path = '/path/to/your/stock_data.db'
# table_names = ['line_sh600000_...']  # 需要处理的表名列表
# for X, y in MultiTableDatasetIterator(db_path, table_names, window_size=30):
#     print(X.shape, y)