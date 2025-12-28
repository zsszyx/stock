from abc import ABC, abstractmethod
import pandas as pd

class BaseRepository(ABC):
    """
    所有数据仓库的抽象基类。
    定义了数据持久化层的通用接口, 如保存和加载。
    """

    @abstractmethod
    def save_data(self, df: pd.DataFrame, table_name: str, **kwargs):
        """
        将一个 DataFrame 保存到持久化存储的指定表中。

        :param df: 要保存的 pandas DataFrame。
        :param table_name: 要写入的目标表的名称。
        :param kwargs: 其他特定于实现的参数 (例如 if_exists)。
        """
        pass

    @abstractmethod
    def load(self, table_name: str, **kwargs) -> pd.DataFrame:
        """
        从持久化存储的指定表中加载数据。

        :param table_name: 要读取的目标表的名称。
        :param kwargs: 其他特定于实现的参数 (例如查询条件)。
        :return: 一个包含所请求数据的 pandas DataFrame。
        """
        pass