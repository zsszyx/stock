import pandas as pd
import os
import sys

# Add project root to Python path to allow importing from other modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from strategy.distribution import calculate_distribution_metrics, calculate_pct_change_metrics
from strategy.filter import apply_length_filter, apply_nan_filter, apply_increase_filter
from sql_op.op import SqlOp
from sql_op import sql_config

def select_stocks_by_skew_kurtosis(df: pd.DataFrame, n_top: int = 100):
    """
    Selects stocks based on highest kurtosis, most left-skewed distribution, and
    most left-skewed pct_chg, after applying all filters.

    Args:
        df (pd.DataFrame): The raw DataFrame with stock data.
        n_top (int): The number of top stocks to select for each criterion.

    Returns:
        tuple: A tuple containing four lists:
               - Stocks with the highest kurtosis.
               - Stocks with the most left-skewed distribution.
               - Stocks with the most left-skewed pct_chg.
               - The intersection of all three lists.
    """
    # Apply all filters first
    df = apply_length_filter(df)
    df = apply_nan_filter(df)
    df = apply_increase_filter(df)
    
    df['pass_all_filters'] = df['length_filter'] & df['nan_filter'] & df['increase_filter']
    filtered_df = df[df['pass_all_filters']].copy()

    if filtered_df.empty:
        return [], [], [], []

    # Calculate distribution metrics for kurtosis and skew
    dist_metrics = calculate_distribution_metrics(filtered_df)
    
    # Calculate pct_change metrics for pct_chg skew
    pct_change_metrics = calculate_pct_change_metrics(filtered_df)

    if dist_metrics.empty or pct_change_metrics.empty:
        return [], [], [], []

    # --- Selection based on distribution metrics ---
    latest_month_dist = dist_metrics['month_index'].min()
    latest_dist_metrics = dist_metrics[dist_metrics['month_index'] == latest_month_dist]

    top_kurtosis_stocks = latest_dist_metrics.sort_values(by='weighted_kurtosis', ascending=False).head(n_top)
    top_skew_stocks = latest_dist_metrics.sort_values(by='weighted_skew', ascending=False).head(n_top)

    # --- Selection based on pct_change metrics ---
    latest_month_pct = pct_change_metrics['month_index'].min()
    latest_pct_metrics = pct_change_metrics[pct_change_metrics['month_index'] == latest_month_pct]
    
    top_pct_chg_skew_stocks = latest_pct_metrics.sort_values(by='weighted_skew_pct_chg', ascending=True).head(n_top)

    kurtosis_list = top_kurtosis_stocks['code'].tolist()
    skew_list = top_skew_stocks['code'].tolist()
    pct_chg_skew_list = top_pct_chg_skew_stocks['code'].tolist()

    # Find the intersection of the three lists
    intersection_stocks = list(set(kurtosis_list) & set(skew_list) & set(pct_chg_skew_list))

    return kurtosis_list, skew_list, pct_chg_skew_list, intersection_stocks

if __name__ == '__main__':
    # --- Example Usage ---
    
    # 1. Initialize database connection
    sql_op = SqlOp(sql_config.db_path)
    
    # 2. Load data for a specific period
    table_name = sql_config.mintues5_table_name
    start_date = '2025-12-01'
    end_date = '2026-01-01'
    print(f"Loading data from {start_date} to {end_date}...")
    k_data = sql_op.read_k_data_by_date_range(table_name, start_date, end_date)

    if not k_data.empty:
        # 3. Select stocks based on the custom strategy
        print("Selecting stocks based on multiple criteria for the latest month...")
        kurtosis_stocks, skew_stocks, pct_chg_skew_stocks, intersection = select_stocks_by_skew_kurtosis(k_data, n_top=100)

        # 4. Print the results
        print(f"\n--- Top 25 Stocks with Highest Kurtosis ---")
        print(f"Found {len(kurtosis_stocks)} stocks.")
        print(kurtosis_stocks)

        print(f"\n--- Top 25 Most Left-Skewed Stocks (Distribution) ---")
        print(f"Found {len(skew_stocks)} stocks.")
        print(skew_stocks)

        print(f"\n--- Top 25 Most Left-Skewed Stocks (Pct Change) ---")
        print(f"Found {len(pct_chg_skew_stocks)} stocks.")
        print(pct_chg_skew_stocks)

        print(f"\n--- Intersection of All Three Lists ---")
        print(f"Found {len(intersection)} stocks in the intersection.")
        print(intersection)
    else:
        print("No data loaded from the database.")

    # 5. Close the database connection
    sql_op.close()