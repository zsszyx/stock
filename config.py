from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    # Base Path: /Users/zsy/stock/stock
    BASE_DIR: Path = Path(__file__).parent.parent
    
    # ClickHouse Configuration
    CH_HOST: str = "localhost"
    CH_PORT: int = 8123
    CH_USER: str = "default"
    CH_PASSWORD: str = ""
    CH_DATABASE: str = "default"

    # Table Names
    TABLE_MIN5: str = "mintues5"
    TABLE_DAILY: str = "daily_kline"
    TABLE_STOCK_LIST: str = "stock_list"
    TABLE_TRADE_DATE: str = "trade_date"
    TABLE_CONCEPT_THS: str = "concept_ths"
    TABLE_CONCEPT_CONSTITUENT_THS: str = "concept_constituent_ths"
    TABLE_BENCHMARK: str = "index_benchmark"

    # Task Settings
    DEFAULT_CHUNK_SIZE: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
