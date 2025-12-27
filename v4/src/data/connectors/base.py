from abc import ABC, abstractmethod
import pandas as pd

class BaseConnector(ABC):
    """
    所有数据连接器的抽象基类 (Abstract Base Class)。
    它定义了所有子类都必须遵守的通用接口。
    """

    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """
        从数据源获取原始数据。
        
        子类必须实现这个方法。
        
        :param kwargs: fetch 方法可能需要的任何特定参数, 例如 file_path, api_key 等。
        :return: 一个包含原始数据的 pandas DataFrame。
        """
        pass