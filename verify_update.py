import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = 'stock.db'
TARGET_DATE = '2026-01-29'
TARGET_DATE_STR = '20260129' # For time field prefix match if needed

def verify_data():
    conn = sqlite3.connect(DB_PATH)
    
    print(f"Analyzing data for date: {TARGET_DATE} ...")
    
    # 1. Define 'Active' Stock Universe
    # We define active stocks as any stock (excluding indices) that has data in the database
    # Since we don't have a stock_list table, we infer from mintues5.
    # To be safe, we check distinct codes present in the DB.
    # If the DB was just created/populated, this is accurate.
    # Filter out 'sh.000' and 'sz.399' (indices)
    
    print("Fetching active stock list from DB...")
    query_universe = """
    SELECT DISTINCT code FROM mintues5 
    WHERE code NOT LIKE 'sh.000%' AND code NOT LIKE 'sz.399%'
    """
    df_universe = pd.read_sql_query(query_universe, conn)
    total_stocks = len(df_universe)
    print(f"Total tracked stocks (universe): {total_stocks}")
    
    if total_stocks == 0:
        print("Error: No stocks found in database.")
        return

    # 2. Get Statistics for Target Date
    print(f"Fetching statistics for {TARGET_DATE}...")
    query_stats = f"""
    SELECT code, count(*) as k_count, max(time) as last_time_str
    FROM mintues5
    WHERE date = '{TARGET_DATE}'
    AND code NOT LIKE 'sh.000%' AND code NOT LIKE 'sz.399%'
    GROUP BY code
    """
    df_stats = pd.read_sql_query(query_stats, conn)
    
    updated_stocks_count = len(df_stats)
    
    # 3. Analyze Coverage
    coverage_pct = (updated_stocks_count / total_stocks) * 100
    
    print("-" * 30)
    print(f"Stocks updated today: {updated_stocks_count} / {total_stocks}")
    print(f"Coverage: {coverage_pct:.2f}%")
    
    # 4. Analyze K-line Completeness
    # Standard day has 48 bars for 5-min data (4 hours * 12 bars/hour)
    # Allow some tolerance? Let's check strict 48 first, then >= 40.
    
    full_day_count = len(df_stats[df_stats['k_count'] == 48])
    near_full_day_count = len(df_stats[df_stats['k_count'] >= 46]) # Tolerance for small missing data
    
    full_day_pct = (full_day_count / total_stocks) * 100 # base on total universe
    near_full_day_pct = (near_full_day_count / total_stocks) * 100
    
    print(f"Stocks with exactly 48 K-lines: {full_day_count} ({full_day_pct:.2f}%)")
    print(f"Stocks with >= 46 K-lines: {near_full_day_count} ({near_full_day_pct:.2f}%)")
    
    # 5. Analyze Last Time (Recency)
    # Check if last_time_str ends with 150000000 (15:00)
    # Format sample: 20251201093500000 -> YYYYMMDDHHMMSSmmm
    # We look for HHMMSS = 150000
    
    def is_market_close(time_str):
        if not time_str or len(time_str) < 14: return False
        # Extract HHMM part. Index 8 to 12.
        # 20260129 1500 00 000
        # 01234567 8901 23 456
        hhmm = time_str[8:12]
        return hhmm == '1500'

    df_stats['is_closed'] = df_stats['last_time_str'].apply(is_market_close)
    closed_count = df_stats['is_closed'].sum()
    closed_pct = (closed_count / total_stocks) * 100
    
    print(f"Stocks with last K-line at 15:00: {closed_count} ({closed_pct:.2f}%)")
    print("-" * 30)
    
    # Final Verdict
    success = True
    if coverage_pct < 90:
        print("FAIL: Coverage is below 90%.")
        success = False
    
    if near_full_day_pct < 90:
        print("FAIL: K-line completeness (>= 46 bars) is below 90%.")
        success = False
        
    if closed_pct < 90:
        print("FAIL: Stocks reaching 15:00 close time is below 90%.")
        success = False

    if success:
        print("SUCCESS: Data update meets all criteria (>90% coverage, completeness, and recency).")
    else:
        print("WARNING: Some criteria were not met.")

    conn.close()

if __name__ == "__main__":
    verify_data()
