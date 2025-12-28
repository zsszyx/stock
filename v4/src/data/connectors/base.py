from abc import ABC, abstractmethod
import pandas as pd

class BaseConnector(ABC):
    """
    所有数据连接器的抽象基类 (Abstract Base Class)。
    它定义了所有子类都必须遵守的通用接口。
    """