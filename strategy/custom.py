import pandas as pd
import os
import sys

# Add project root to Python path to allow importing from other modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from strategy.distribution import calculate_distribution_metrics
from sql_op.op import SqlOp
from sql_op import sql_config

def select_stocks_by_skew_kurtosis(distribution_metrics_df, n_top=100):
    """
    Selects stocks based on the highest kurtosis and the most left-skewed (lowest skewness).

    Args:
        distribution_metrics_df (pd.DataFrame): DataFrame containing distribution metrics
                                                per stock, must include 'code', 'month_index',
                                                'weighted_kurtosis', and 'weighted_skew'.
        n_top (int): The number of top stocks to select for each criterion.

    Returns:
        tuple: A tuple containing three lists:
               - A list of stock codes with the highest kurtosis.
               - A list of stock codes with the lowest (most left-skewed) skewness.
               - A list of stock codes that are in both of the above lists (intersection).
    """
    if distribution_metrics_df.empty:
        return [], [], []

    # We only care about the most recent month's metrics for selection
    latest_month_index = distribution_metrics_df['month_index'].min()
    latest_metrics = distribution_metrics_df[distribution_metrics_df['month_index'] == latest_month_index]

    # Sort by kurtosis (descending) and select top N
    top_kurtosis_stocks = latest_metrics.sort_values(by='weighted_kurtosis', ascending=False).head(n_top)
    
    # Sort by skewness (ascending) to find the most left-skewed and select top N
    top_skew_stocks = latest_metrics.sort_values(by='weighted_skew', ascending=False).head(n_top)

    kurtosis_list = top_kurtosis_stocks['code'].tolist()
    skew_list = top_skew_stocks['code'].tolist()

    # Find the intersection of the two lists
    intersection_stocks = list(set(kurtosis_list) & set(skew_list))

    return kurtosis_list, skew_list, intersection_stocks

if __name__ == '__main__':
    # --- Example Usage ---
    
    # 1. Initialize database connection
    sql_op = SqlOp(sql_config.db_path)
    
    # 2. Load data for a specific period
    table_name = sql_config.mintues5_table_name
    start_date = '2025-10-01'
    end_date = '2026-01-01'
    print(f"Loading data from {start_date} to {end_date}...")
    k_data = sql_op.read_k_data_by_date_range(table_name, start_date, end_date)

    if not k_data.empty:
        # 3. Calculate monthly distribution metrics
        print("Calculating monthly distribution metrics...")
        dist_metrics = calculate_distribution_metrics(k_data.copy())

        if not dist_metrics.empty:
            # 4. Select stocks based on the custom strategy
            print("Selecting stocks based on skew and kurtosis for the latest month...")
            highest_kurtosis_stocks, most_left_skewed_stocks, intersection = select_stocks_by_skew_kurtosis(dist_metrics, n_top=25)

            # 5. Print the results
            print(f"\n--- Top 200 Stocks with Highest Kurtosis ---")
            print(f"Found {len(highest_kurtosis_stocks)} stocks.")
            print(highest_kurtosis_stocks)

            print(f"\n--- Top 200 Most Left-Skewed Stocks ---")
            print(f"Found {len(most_left_skewed_stocks)} stocks.")
            print(most_left_skewed_stocks)

            print(f"\n--- Intersection of Both Lists ---")
            print(f"Found {len(intersection)} stocks in the intersection.")
            print(intersection)
        else:
            print("Distribution metrics calculation resulted in an empty DataFrame.")
    else:
        print("No data loaded from the database.")

    # 6. Close the database connection
    sql_op.close()