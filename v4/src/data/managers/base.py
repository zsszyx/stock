from abc import ABC, abstractmethod
from src.data.connectors.base import BaseConnector
from src.data.repositories.base import BaseRepository

class BaseManager(ABC):
    """
    所有数据管理器的抽象基类。
    它定义了数据管理器的通用接口，负责协调 Connector 和 Repository，完成数据的同步。
    """

    def __init__(self, connector: BaseConnector, repository: BaseRepository):
        """
        初始化 BaseManager。

        :param connector: 一个 BaseConnector 的实例，用于从数据源获取数据。
        :param repository: 一个 BaseRepository 的实例，用于将数据持久化。
        """
        self.connector = connector
        self.repository = repository

    @abstractmethod
    def sync_data(self, **kwargs):
        """
        同步数据的抽象方法。
        
        子类必须实现这个方法，定义具体的数据同步逻辑，
        例如：从哪里获取数据、获取什么时间范围的数据、以及如何存储数据。
        """
        pass