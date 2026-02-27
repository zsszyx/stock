from stock.database.factory import RepositoryFactory
from stock.config import settings
import pandas as pd

def debug_ksp():
    repo = RepositoryFactory.get_clickhouse_repo()
    dates = ['2025-01-02', '2025-02-03', '2025-03-03']
    
    for d in dates:
        print(f"\n--- Checking date: {d} ---")
        query = f"""
        SELECT 
            count(*) as total,
            countIf(ksp_sum_5d != 0) as has_5d,
            countIf(ksp_sum_5d_rank != 0) as has_5d_rank,
            avg(ksp_sum_5d) as avg_5d,
            min(ksp_sum_5d_rank) as min_rank,
            max(ksp_sum_5d_rank) as max_rank
        FROM {settings.TABLE_DAILY}
        WHERE date = '{d}'
        """
        res = repo.query(query)
        print(res)
        
        sample_query = f"""
        SELECT code, ksp_sum_5d, ksp_sum_5d_rank
        FROM {settings.TABLE_DAILY}
        WHERE date = '{d}' AND ksp_sum_5d_rank > 0
        ORDER BY ksp_sum_5d_rank ASC
        LIMIT 5
        """
        samples = repo.query(sample_query)
        print("Top 5 samples:")
        print(samples)

if __name__ == "__main__":
    debug_ksp()
