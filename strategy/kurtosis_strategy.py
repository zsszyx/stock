import datetime
import pandas as pd
import sys
import os
# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sql_op import op, sql_config
from strategy.filter import apply_volume_zero_filter

def weighted_kurtosis(values, weights):
    """
    Computes the weighted kurtosis.
    """
    if weights.sum() == 0:
        return 0
    mean = (values * weights).sum() / weights.sum()
    centered_values = values - mean
    variance = (centered_values**2 * weights).sum() / weights.sum()
    if variance == 0:
        return 0
    std_dev = variance**0.5
    
    m4 = (centered_values**4 * weights).sum() / weights.sum()
    
    return m4 / (std_dev**4) - 3

def run_kurtosis_strategy():
    """
    This strategy identifies the top 3 concepts with the highest average kurtosis
    and the top 5 stocks with the highest kurtosis within each of those concepts.
    """
    sql_operator = op.SqlOp()
    
    # 1. Get the last 5 days of data
    # Get actual available dates from DB to avoid empty result on weekends/holidays
    date_query = f"SELECT DISTINCT date FROM {sql_config.mintues5_table_name} ORDER BY date DESC LIMIT 10"
    dates_df = sql_operator.query(date_query)
    1
    if dates_df is None or dates_df.empty:
        print("No data found in database.")
        return

    dates = sorted(dates_df['date'].astype(str).tolist())
    start_date = dates[0]
    end_date = dates[-1]
    print(f"Analyzing data from {start_date} to {end_date}")
    
    k_data = sql_operator.read_k_data_by_date_range(sql_config.mintues5_table_name, start_date, end_date)
    
    if k_data.empty:
        print("No data found for the last 5 days.")
        return

    # Normalize stock codes (remove sh./sz. prefix)
    k_data['code'] = k_data['code'].astype(str).apply(lambda x: x.split('.')[-1])

    # 2. Calculate price and weighted kurtosis for each stock
    k_data = apply_volume_zero_filter(k_data)
    k_data = k_data[k_data['volume_filter']]
    k_data.loc[:, 'amount'] = pd.to_numeric(k_data['amount'], errors='coerce')
    k_data['price'] = k_data['amount'] / k_data['volume']
    
    stock_kurtosis = {}
    for code, group in k_data.groupby('code'):
        if group['volume'].sum() > 0:
            kurt = weighted_kurtosis(group['price'], group['volume'])
            stock_kurtosis[code] = kurt

    if not stock_kurtosis:
        print("Could not calculate kurtosis for any stock.")
        return

    # 3. Get concept constituents
    concept_constituents = sql_operator.read_concept_constituent()
    if concept_constituents.empty:
        print("No concept constituent data found.")
        return

    # 4. Calculate average kurtosis for each concept
    concept_kurtosis = {}
    for concept_code, group in concept_constituents.groupby('concept_code'):
        concept_stocks = group['code'].tolist()
        kurtosis_values = [stock_kurtosis.get(stock_code) for stock_code in concept_stocks if stock_code in stock_kurtosis]
        if kurtosis_values:
            average_kurtosis = sum(kurtosis_values) / len(kurtosis_values)
            concept_kurtosis[concept_code] = average_kurtosis

    if not concept_kurtosis:
        print("Could not calculate average kurtosis for any concept.")
        return

    # 5. Get top 3 concepts with the highest average kurtosis
    sorted_concepts = sorted(concept_kurtosis.items(), key=lambda item: item[1], reverse=True)
    top_3_concepts = sorted_concepts[:3]

    print("Top 3 Concepts by Average Kurtosis:")
    for concept_code, avg_kurt in top_3_concepts:
        # Correct column name is 'concept', not 'concept_name'
        concept_name = concept_constituents[concept_constituents['concept_code'] == concept_code]['concept'].iloc[0]
        print(f"\nConcept: {concept_name} (Code: {concept_code}), Average Kurtosis: {avg_kurt:.2f}")
        
        # 6. Get top 5 stocks in this concept by kurtosis
        concept_stocks = concept_constituents[concept_constituents['concept_code'] == concept_code]['code'].tolist()
        
        stock_kurtosis_in_concept = {code: stock_kurtosis.get(code, 0) for code in concept_stocks}
        
        sorted_stocks = sorted(stock_kurtosis_in_concept.items(), key=lambda item: item[1], reverse=True)
        top_5_stocks = sorted_stocks[:5]
        
        print("  Top 5 Stocks by Kurtosis:")
        for stock_code, kurt in top_5_stocks:
            print(f"    - Stock Code: {stock_code}, Kurtosis: {kurt:.2f}")

if __name__ == '__main__':
    run_kurtosis_strategy()
