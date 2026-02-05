import pandas as pd
import sqlite3
from datetime import datetime
from stock.data_context import Minutes5Context, DailyContext, ConceptContext
from stock.sql_op import sql_config

def run_concept_analysis():
    # 1. Setup Database and fetch data
    db_file_path = sql_config.db_path.replace("sqlite:///", "")
    with sqlite3.connect(db_file_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT MAX(date) FROM {sql_config.mintues5_table_name}")
        latest_date_str = cursor.fetchone()[0]
    
    if not latest_date_str:
        print("No data found in database.")
        return

    print(f"Analysis for date: {latest_date_str}")

    # Fetch last 15 days
    query = f"""
    SELECT * FROM {sql_config.mintues5_table_name} 
    WHERE date IN (
        SELECT DISTINCT date FROM {sql_config.mintues5_table_name} 
        ORDER BY date DESC LIMIT 15
    )
    """
    print("Fetching 5-minute data from SQL...")
    with sqlite3.connect(db_file_path) as conn:
        df = pd.read_sql_query(query, conn)
    
    # 2. Initialize Contexts
    print("Initializing Contexts and calculating daily stats...")
    min5_ctx = Minutes5Context(df)
    daily_ctx = DailyContext(min5_ctx)
    concept_ctx = ConceptContext()
    
    # 3. Get latest day stats
    latest_daily = daily_ctx.data[daily_ctx.data['date'] == latest_date_str].copy()
    
    # 4. Merge with Concept Data
    merged_df = latest_daily.merge(concept_ctx._df, on='code')
    
    # 5. Aggregate by Concept
    print("Aggregating statistics by concept...")
    concept_agg = merged_df.groupby('concept').agg({
        'amplitude': 'mean',
        'skew': 'mean',
        'kurt': 'mean'
    }).reset_index()
    
    # 6. Filter: Mean Amplitude < 5% (0.05) and Mean Skew < 0
    mask = (concept_agg['amplitude'] < 0.05) & (concept_agg['skew'] < 0)
    filtered_concepts = concept_agg[mask].sort_values('kurt', ascending=False)
    
    if filtered_concepts.empty:
        print("\nNo concepts matched strict criteria (Amp < 5% and Skew < 0).")
        return

    # 7. Iterate through top 3 concepts
    top_n = min(3, len(filtered_concepts))
    print(f"\n--- Top {top_n} Concepts Found ---")
    
    for i in range(top_n):
        top_concept_row = filtered_concepts.iloc[i]
        top_concept_name = top_concept_row['concept']
        
        print(f"\nRank {i+1}: {top_concept_name}")
        print(f"  Mean Amplitude: {top_concept_row['amplitude']:.2%}")
        print(f"  Mean Skewness: {top_concept_row['skew']:.4f}")
        print(f"  Mean Kurtosis: {top_concept_row['kurt']:.4f}")
        
        # 8. List stocks in this concept
        stocks_in_concept = concept_ctx.get_stocks(top_concept_name)
        active_stocks = [s for s in stocks_in_concept if s in latest_daily['code'].values]
        
        concept_stocks_df = latest_daily[latest_daily['code'].isin(active_stocks)].sort_values('kurt', ascending=False)
        print(f"  Stocks in '{top_concept_name}' ({len(active_stocks)} active):")
        # Print top 5 stocks by kurtosis for this concept
        print(concept_stocks_df[['code', 'close', 'amplitude', 'skew', 'kurt']].head(5).to_string(index=False))

if __name__ == "__main__":
    run_concept_analysis()
