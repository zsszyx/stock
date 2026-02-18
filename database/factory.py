from stock.database.repositories.clickhouse_repo import ClickHouseRepository
from stock.database.repositories.sqlite_repo import SQLiteRepository
from stock.database.base import BaseRepository

class RepositoryFactory:
    @staticmethod
    def get_clickhouse_repo() -> ClickHouseRepository:
        return ClickHouseRepository()

    @staticmethod
    def get_sqlite_repo() -> SQLiteRepository:
        return SQLiteRepository()
