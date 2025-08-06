import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from preprocess import MultiTableDatasetIterator

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

# 用法示例
# db_path = '/path/to/your/stock_data.db'
# table_names = [...]
# dataset = MultiTableTorchIterableDataset(db_path, table_names, window_size=30)
# loader = DataLoader(dataset, batch_size=64)  # shuffle参数无效
# for X, y in loader:
#     print(X.shape, y.shape)
