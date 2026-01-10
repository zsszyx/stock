import sys
import os
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sql_op.op import SqlOp
from strategy.distribution import calculate_rolling_distribution_divergence
from sql_op import sql_config

def find_smallest_divergence_stocks(top_n: int = 10):
    """
    Finds the top N stocks with the smallest KL divergence for the most recent date in the data.

    Args:
        top_n (int): The number of stocks to return.

    Returns:
        pd.DataFrame: A DataFrame containing the top N stocks with the smallest divergence.
    """
    sql_op = SqlOp()
    
    # Read all k-line data
    k_data = sql_op.read_all_k_data(sql_config.mintues5_table_name)

    if k_data is None or k_data.empty:
        print("No data available.")
        sql_op.close()
        return pd.DataFrame()

    # Determine the latest date from the data
    latest_date = k_data['date'].max().strftime('%Y-%m-%d')
    print(f"Latest date found in data: {latest_date}")

    # To calculate divergence for a date, we need data from the preceding days.
    # The window size in calculate_rolling_distribution_divergence is 5 by default, 
    # and it compares [t-5, t] with [t-6, t-1]. So we need at least 7 days of data.
    # Let's filter the data for a slightly larger period to be safe.
    start_date = (pd.to_datetime(latest_date) - pd.Timedelta(days=15)).strftime('%Y-%m-%d')
    
    k_data_filtered = k_data[(k_data['date'] >= start_date) & (k_data['date'] <= latest_date)]

    print("Calculating rolling distribution divergence...")
    divergence_results = calculate_rolling_distribution_divergence(k_data_filtered)

    if divergence_results.empty:
        print("Could not calculate rolling divergence.")
        sql_op.close()
        return pd.DataFrame()

    # Filter for the specific date
    divergence_for_date = divergence_results[divergence_results['date'].dt.strftime('%Y-%m-%d') == latest_date]

    if divergence_for_date.empty:
        print(f"No divergence results for the date {latest_date}")
        sql_op.close()
        return pd.DataFrame()

    # Sort by KL divergence in ascending order and take the top N
    smallest_divergence_stocks = divergence_for_date.sort_values('kl_divergence', ascending=True).head(top_n)

    sql_op.close()
    return smallest_divergence_stocks

if __name__ == '__main__':
    print("Finding stocks with smallest divergence for the latest date...")
    top_10_stocks = find_smallest_divergence_stocks()
    
    if not top_10_stocks.empty:
        print("\nTop 10 stocks with the smallest KL divergence:")
        print(top_10_stocks)
    else:
        print("Could not find any stocks with the smallest divergence.")