from stock.tasks.base import BaseTask
from stock.database.factory import RepositoryFactory
from stock.config import settings

class ConceptMigrationTask(BaseTask):
    def __init__(self):
        super().__init__("ConceptMigrationTask")
        self.sqlite_repo = RepositoryFactory.get_sqlite_repo()
        self.ch_repo = RepositoryFactory.get_clickhouse_repo()

    def run(self):
        self.log_progress("Starting Concept Data Migration: SQLite -> ClickHouse")
        
        # 1. Ensure CH table exists
        self.ch_repo.create_concept_tables()
        
        # 2. Read from SQLite
        query = f"SELECT code, concept FROM {settings.TABLE_CONCEPT_CONSTITUENT_THS}"
        df = self.sqlite_repo.query(query)
        
        if df.empty:
            self.log_progress("No concept data found in SQLite.")
            return

        self.log_progress(f"Migrating {len(df)} concept-stock mapping records...")
        
        # 3. Insert into ClickHouse
        # Note: We keep raw codes here, ConceptContext handles prefixes
        self.ch_repo.insert_df(df, settings.TABLE_CONCEPT_CONSTITUENT_THS)
        
        # 4. Physical deduplication
        self.ch_repo.optimize_table(settings.TABLE_CONCEPT_CONSTITUENT_THS)
        
        self.log_progress("Migration and optimization completed.")

    def close(self):
        self.sqlite_repo.close()
        self.ch_repo.close()
