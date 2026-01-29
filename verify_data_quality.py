import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

DB_PATH = 'stock.db'
TABLE_NAME = 'mintues5'

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def analyze_data_quality():
    conn = get_db_connection()
    
    print("Initializing Data Quality Check...")
    
    # 1. Get Global Statistics
    print("Fetching global metadata...")
    # Get all distinct codes (Universe) excluding indices
    query_codes = f"SELECT DISTINCT code FROM {TABLE_NAME} WHERE code NOT LIKE 'sh.000%' AND code NOT LIKE 'sz.399%'"
    all_codes = pd.read_sql_query(query_codes, conn)['code'].tolist()
    total_codes = len(all_codes)
    print(f"Total Unique Stocks Found: {total_codes}")

    # Get all distinct dates
    query_dates = f"SELECT DISTINCT date FROM {TABLE_NAME} ORDER BY date"
    all_dates = pd.read_sql_query(query_dates, conn)['date'].tolist()
    print(f"Total Trading Days Found: {len(all_dates)}")
    print(f"Date Range: {all_dates[0]} to {all_dates[-1]}")
    
    # Check Date Continuity (Simple check for large gaps)
    # Convert to datetime to check gaps > wait, weekends exist. 
    # Simply listing dates with < 5000 stocks might be easier to spot missing days if we assume partial updates.
    
    print("\n" + "="*50)
    print(f"{ 'DATE':<12} | {'COVERAGE':<10} | {'PERFECT':<10} | {'MISSING':<8} | {'ABNORMAL':<8} | {'NANs':<5}")
    print("="*50)
    
    issues_report = []
    
    for date in tqdm(all_dates, desc="Scanning Dates"):
        # Query stats for this specific date
        # We aggregate in SQL for performance
        query = f"""
        SELECT 
            code,
            count(*) as k_count,
            min(time) as min_time,
            max(time) as max_time,
            sum(CASE 
                WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL OR amount IS NULL 
                THEN 1 ELSE 0 
            END) as null_rows
        FROM {TABLE_NAME}
        WHERE date = '{date}'
        AND code NOT LIKE 'sh.000%' AND code NOT LIKE 'sz.399%'
        GROUP BY code
        """
        
        df_day = pd.read_sql_query(query, conn)
        
        # Metrics
        day_codes_count = len(df_day)
        coverage_pct = (day_codes_count / total_codes) * 100
        
        # 1. Check Completeness (48 bars)
        # Note: Some stocks might be halted for an hour, resulting in e.g. 40 bars. 
        # But for 'Perfect' status we demand 48.
        perfect_count_mask = (df_day['k_count'] == 48)
        
        # 2. Check Time Boundaries (Start 09:35, End 15:00)
        # time format: YYYYMMDDHHMMSSmmm
        # We check substring. 
        # min should end in 093500000, max should end in 150000000
        def check_time(row):
            if not isinstance(row['min_time'], str) or not isinstance(row['max_time'], str):
                return False
            start_ok = row['min_time'][8:12] == '0935'
            end_ok = row['max_time'][8:12] == '1500'
            return start_ok and end_ok

        time_ok_mask = df_day.apply(check_time, axis=1)
        
        # 3. Check NaNs
        no_nan_mask = (df_day['null_rows'] == 0)
        
        # Combine for Perfect Score
        is_perfect = perfect_count_mask & time_ok_mask & no_nan_mask
        perfect_stocks_count = is_perfect.sum()
        
        # Analyze Abnormalities
        abnormal_mask = ~is_perfect
        abnormal_stocks = df_day[abnormal_mask]
        abnormal_count = len(abnormal_stocks)
        
        nan_issues = df_day['null_rows'].sum()
        
        # Record specific issues for this date
        if abnormal_count > 0:
            # Categorize issues
            cnt_issue = (~perfect_count_mask).sum()
            time_issue = (~time_ok_mask).sum()
            nan_issue = (~no_nan_mask).sum()
            
            # If significant issues, add to detailed report
            # We define significant as > 1% of active stocks having issues
            if abnormal_count > (day_codes_count * 0.01):
                issues_report.append({
                    'date': date,
                    'abnormal_count': abnormal_count,
                    'count_mismatch': cnt_issue,
                    'time_mismatch': time_issue,
                    'nan_found': nan_issue
                })

        # Print Row
        # Format: Date | Coverage% | Perfect% | Missing | Abnormal | NaNs
        print(f"{date:<12} | {coverage_pct:6.2f}%    | {(perfect_stocks_count/day_codes_count)*100:6.2f}%    | {total_codes - day_codes_count:<8} | {abnormal_count:<8} | {nan_issues:<5}")

    print("="*50)
    
    if issues_report:
        print("\n[!] DETAILED ISSUES FOUND:")
        for rep in issues_report:
            print(f"Date: {rep['date']}")
            print(f"  - Abnormal Stocks: {rep['abnormal_count']}")
            print(f"  - K-Line Count Mismatch (!=48): {rep['count_mismatch']}")
            print(f"  - Market Open/Close Time Mismatch: {rep['time_mismatch']}")
            print(f"  - Rows with NaNs: {rep['nan_found']}")
            print("-" * 20)
    else:
        print("\n[+] No significant data anomalies detected across all dates.")
        
    conn.close()

if __name__ == "__main__":
    analyze_data_quality()
