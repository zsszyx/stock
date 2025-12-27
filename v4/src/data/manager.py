import pandas as pd
from .connectors.base import BaseConnector
from .repositories.base import BaseRepository
from ..common.schema import transform_dataframe

class DataManager:
    """
    数据管理层, 负责协调数据获取、转换和存储的整个流程。
    """

    def __init__(self, connector: BaseConnector, repository: BaseRepository):
        """
        初始化 DataManager。

        :param connector: 一个遵循 BaseConnector 接口的数据连接器实例。
        :param repository: 一个遵循 BaseRepository 接口的数据仓库实例。
        """
        self.connector = connector
        self.repository = repository

    def etl_and_save(self, fetch_kwargs: dict, save_kwargs: dict):
        """
        执行完整的 ETL (Extract, Transform, Load) 流程并保存数据。

        1. 从连接器 (Connector) 提取 (Extract) 原始数据。
        2. 使用 common.schema 中的函数转换 (Transform) 数据, 使其标准化。
        3. 通过仓库 (Repository) 加载 (Load), 即保存到持久化存储中。

        :param fetch_kwargs: 传递给 connector.fetch 方法的参数字典。
        :param save_kwargs: 传递给 repository.save 方法的参数字典。
        """
        print("开始从数据源获取数据...")
        raw_df = self.connector.fetch(**fetch_kwargs)
        
        print("数据获取完毕, 开始进行标准化转换...")
        transformed_df = transform_dataframe(raw_df)
        
        print("数据转换完毕, 开始将其保存到仓库中...")
        self.repository.save(transformed_df, **save_kwargs)
        
        print("数据处理和保存流程全部完成!")

    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        从仓库中加载数据。

        :param kwargs: 传递给 repository.load 方法的参数字典。
        :return: 加载到的数据。
        """
        print(f"正在从仓库加载数据, 参数: {kwargs}")
        return self.repository.load(**kwargs)