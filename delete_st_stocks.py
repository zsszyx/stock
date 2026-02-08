import sys
import os
import pandas as pd
from tqdm import tqdm

# Add root to path
sys.path.append(os.getcwd())

from data_fetch.data_provider.baostock_provider import BaoInterface
from sql_op.op import SqlOp
from sql_op import sql_config

def delete_st_stocks():
    print("Initializing ST cleanup...")
    sql_op = SqlOp()
    
    # 1. Get current stock list from Baostock to identify ST stocks
    print("Fetching current stock list from Baostock...")
    with BaoInterface() as bi:
        # Get latest trading date or just use a recent one
        # For simplicity, we can use the last date from our DB or just today's date
        # Baostock's query_all_stock needs a valid date. 
        # Let's get the max date from our DB.
        res = sql_op.query(f"SELECT MAX(date) as max_date FROM {sql_config.mintues5_table_name}")
        max_date = res.iloc[0]['max_date'] if not res.empty else "2026-02-06"
        
        stock_list = bi.get_stock_list(date=max_date)
        
    if stock_list is None or stock_list.empty:
        print("Failed to fetch stock list from Baostock.")
        return

    # 2. Filter for ST, *ST, S, T stocks
    # Using the same logic found in update_min5_task.py
    st_filter = (
        (stock_list['code_name'].str.contains('ST', case=False, na=False)) |
        (stock_list['code_name'].str.contains('\*ST', case=False, na=False)) |
        (stock_list['code_name'].str.contains('S', case=False, na=False)) |
        (stock_list['code_name'].str.contains('T', case=False, na=False))
    )
    st_stocks = stock_list[st_filter]
    st_codes = st_stocks['code'].tolist()
    
    print(f"Found {len(st_codes)} ST/Risk stocks to delete.")
    
    if not st_codes:
        print("No ST stocks found.")
        return

    # 3. Delete from database
    chunk_size = 500
    total_deleted = 0
    for i in tqdm(range(0, len(st_codes), chunk_size), desc="Deleting ST stocks"):
        batch = st_codes[i:i+chunk_size]
        codes_str = "', '".join(batch)
        query = f"DELETE FROM {sql_config.mintues5_table_name} WHERE code IN ('{codes_str}')"
        rows = sql_op.execute_non_query(query)
        if rows > 0:
            total_deleted += rows

    print(f"Cleanup complete. Total rows deleted: {total_deleted}")

if __name__ == "__main__":
    delete_st_stocks()
