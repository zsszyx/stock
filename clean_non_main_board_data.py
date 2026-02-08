import sys
import os
import re
from tqdm import tqdm
import time

# Add root to path
sys.path.append(os.getcwd())

from sql_op.op import SqlOp
from sql_op import sql_config

def clean_data():
    print("Initializing cleanup...")
    sql_op = SqlOp()
    
    print("Executing direct SQL DELETE for non-Main Board/ChiNext stocks...")
    print("Keeping only: sh.60%, sz.00%, sz.30%")
    
    # Construct SQL
    # DELETE FROM table WHERE code NOT LIKE ...
    query = f"""
        DELETE FROM {sql_config.mintues5_table_name} 
        WHERE code NOT LIKE 'sh.60%' 
        AND code NOT LIKE 'sz.00%' 
        AND code NOT LIKE 'sz.30%'
    """
    
    start_time = time.time()
    rows_affected = sql_op.execute_non_query(query)
    end_time = time.time()
    
    print(f"Cleanup complete in {end_time - start_time:.2f} seconds.")
    print(f"Rows deleted: {rows_affected}")
    
    # Optional: Vacuum to reclaim space (User didn't explicitly ask, but good for DB health)
    # print("Vacuuming database (this may take a while)...")
    # sql_op.execute_non_query("VACUUM")
    # print("Vacuum complete.")

if __name__ == "__main__":
    clean_data()
