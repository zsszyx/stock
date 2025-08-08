import random
import sqlite3
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import sys
sys.path.append('/Users/zsy/stock/stock/v2')
from bao_data import DB_PATH
from bao_data.prepare import get_table_names_with_connection
from preprocess import dataset_pipeline, feature_fields


class MultiTableTorchIterableDataset(IterableDataset):
    def __init__(self, db_path, table_names, window_size=30, label_field='label', filter_field='filter'):
        """
        :param db_path: 数据库路径
        :param table_names: 表名列表
        :param window_size: 滑动窗口长度
        :param label_field: 标签字段名
        :param filter_field: 过滤字段名
        """
        self.db_path = db_path
        self.table_names = table_names
        self.window_size = window_size
        self.label_field = label_field
        self.filter_field = filter_field

    def __iter__(self):
        iterator = MultiTableDatasetIterator(
            self.db_path, self.table_names, window_size=self.window_size,
            label_field=self.label_field, filter_field=self.filter_field
        )
        for X, y in iterator:
            yield torch.from_numpy(X.astype(np.float32)), torch.tensor(y, dtype=torch.float32)

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
            X = window[feature_fields[1:]].values
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

if __name__ == "__main__":
    
    table_names = get_table_names_with_connection()
    dataset = MultiTableTorchIterableDataset(DB_PATH, table_names, window_size=30)
    dataloader = DataLoader(dataset, batch_size=32)
    for batch_X, batch_y in dataloader:
        # print(batch_X.shape, batch_y.shape)
        print(batch_y.sum())
        # 这里可以添加模型训练或验证的代码



