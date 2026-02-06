import pandas as pd
import sqlite3
from datetime import datetime
from data_context import Minutes5Context, DailyContext, ConceptContext
from sql_op import sql_config

def test_repetition():
    # 1. Setup
    db_file_path = sql_config.db_path.replace("sqlite:///", "")
    with sqlite3.connect(db_file_path) as conn:
        cursor = conn.cursor()
        # Get last 10 distinct dates
        cursor.execute(f"SELECT DISTINCT date FROM {sql_config.mintues5_table_name} ORDER BY date DESC LIMIT 10")
        dates = [r[0] for r in cursor.fetchall()]
    
    if len(dates) < 5:
        print("Not enough data to test repetition.")
        return

    # We'll test the last 5 days
    test_dates = sorted(dates[:5])
    
    # Fetch data for these days + some buffer for prev_close
    query_dates = dates[:10] # Use more dates for safe aggregation
    query = f"""
    SELECT * FROM {sql_config.mintues5_table_name} 
    WHERE date IN ({','.join(['?' for _ in query_dates])})
    """
    
    print(f"Fetching data for dates: {test_dates}")
    with sqlite3.connect(db_file_path) as conn:
        df = pd.read_sql_query(query, conn, params=query_dates)
    
    # 2. Aggregation
    min5_ctx = Minutes5Context(df)
    daily_ctx = DailyContext(min5_ctx)
    concept_ctx = ConceptContext()
    
    results = {}
    
    for date_str in test_dates:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        latest_daily = daily_ctx.data[daily_ctx.data['date'] == date_str].copy()
        
        if latest_daily.empty:
            continue
            
        merged_df = latest_daily.merge(concept_ctx._df, on='code')
        concept_agg = merged_df.groupby('concept').agg({
            'amplitude': 'mean',
            'skew': 'mean',
            'kurt': 'mean'
        }).reset_index()
        
        mask = (concept_agg['amplitude'] < 0.05) & (concept_agg['skew'] < 0)
        filtered = concept_agg[mask].sort_values('kurt', ascending=False)
        
        top_3 = filtered.head(3)['concept'].tolist()
        results[date_str] = top_3
        print(f"Date {date_str}: Top 3 Concepts -> {top_3}")

    # 3. Analyze Repetition
    print("\n--- Repetition Analysis ---")
    all_selected = []
    for d in results:
        all_selected.extend(results[d])
    
    freq = pd.Series(all_selected).value_counts()
    repeated = freq[freq > 1]
    
    if repeated.empty:
        print("No concepts repeated in the top 3 over the last 5 days.")
    else:
        print("Repeated Concepts in Top 3:")
        for cp, count in repeated.items():
            print(f"  - {cp}: appeared {count} times")

if __name__ == "__main__":
    test_repetition()
