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


def select_stocks_by_weekly_metrics(df: pd.DataFrame, n_top: int = 200):
    """
    Selects stocks based on weekly metrics for real_price and pct_chg.

    Args:
        df (pd.DataFrame): Raw DataFrame with stock data.
        n_top (int): The number of top stocks to select for each criterion.

    Returns:
        tuple: A tuple containing the intersection list and the four individual lists.
    """
    from strategy.period import add_week_index
    import numpy as np

    df.loc[:, 'volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df.loc[:, 'amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df.loc[:, 'close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(subset=['volume', 'amount', 'close'], inplace=True)
    df = df[df['volume'] > 0].copy()
    # Apply all filters first
    df = apply_length_filter(df)
    df = apply_nan_filter(df)
    # df = apply_increase_filter(df)
    
    df['pass_all_filters'] = df['length_filter'] & df['nan_filter']
    df = df[df['pass_all_filters']].copy()

    if df.empty:
        return [], [], [], [], []

    df = df.sort_values(by=['code', 'time'])
    df.loc[:, 'real_price'] = df['amount'] / df['volume']
    df.loc[:, 'pct_chg'] = df.groupby('code')['close'].pct_change()
    df.dropna(subset=['pct_chg'], inplace=True)

    df = add_week_index(df)
    latest_week_df = df[df['week_index'] == 0].copy()

    if latest_week_df.empty:
        return [], [], [], [], []

    def _calculate_metrics(group, value_col):
        volume = group['volume']
        value = group[value_col]
        
        if volume.sum() == 0:
            return pd.Series({'kurtosis': np.nan, 'skew': np.nan})

        weighted_mean = np.average(value, weights=volume)
        weighted_variance = np.average((value - weighted_mean)**2, weights=volume)
        weighted_std = np.sqrt(weighted_variance)

        if weighted_std == 0:
            skew = 0
            kurtosis = 0
        else:
            third_moment = np.average((value - weighted_mean)**3, weights=volume)
            skew = third_moment / (weighted_std**3)
            fourth_moment = np.average((value - weighted_mean)**4, weights=volume)
            kurtosis = (fourth_moment / (weighted_std**4)) - 3
        
        return pd.Series({'kurtosis': kurtosis, 'skew': skew})

    price_metrics = latest_week_df.groupby('code').apply(_calculate_metrics, value_col='real_price', include_groups=False).reset_index()
    price_metrics.rename(columns={'kurtosis': 'kurtosis_price', 'skew': 'skew_price'}, inplace=True)

    pct_chg_metrics = latest_week_df.groupby('code').apply(_calculate_metrics, value_col='pct_chg', include_groups=False).reset_index()
    pct_chg_metrics.rename(columns={'kurtosis': 'kurtosis_pct_chg', 'skew': 'skew_pct_chg'}, inplace=True)

    # 1. Highest Kurtosis (real_price) from stocks with non-negative skew
    top_kurt_price = price_metrics[price_metrics['skew_price'] >= 0].nlargest(n_top, 'kurtosis_price')['code'].tolist()

    # 2. Lowest Absolute Skew (real_price)
    price_metrics['abs_skew_price'] = price_metrics['skew_price'].abs()
    bottom_skew_price = price_metrics.nsmallest(n_top, 'abs_skew_price')['code'].tolist()

    # 3. Highest Kurtosis (pct_chg) from stocks with non-negative skew
    top_kurt_pct_chg = pct_chg_metrics[pct_chg_metrics['skew_pct_chg'] >= 0].nlargest(n_top, 'kurtosis_pct_chg')['code'].tolist()

    # 4. Lowest Absolute Skew (pct_chg)
    pct_chg_metrics['abs_skew_pct_chg'] = pct_chg_metrics['skew_pct_chg'].abs()
    bottom_skew_pct_chg = pct_chg_metrics.nsmallest(n_top, 'abs_skew_pct_chg')['code'].tolist()

    intersection = list(set(top_kurt_price))

    return intersection, top_kurt_price, bottom_skew_price, top_kurt_pct_chg, bottom_skew_pct_chg


if __name__ == '__main__':
    # --- Example Usage ---
    
    # 1. Initialize database connection
    sql_op = SqlOp(sql_config.db_path)
    
    # 2. Load data for a specific period
    table_name = sql_config.mintues5_table_name
    start_date = '2026-01-01'
    end_date = '2026-01-09'
    print(f"Loading data from {start_date} to {end_date}...")
    k_data = sql_op.read_k_data_by_date_range(table_name, start_date, end_date)

    if not k_data.empty:
        # 3. Select stocks based on the new weekly strategy
        print("\nSelecting stocks based on weekly metrics...")
        intersection, kurt_price, skew_price, kurt_pct, skew_pct = select_stocks_by_weekly_metrics(k_data, n_top=50)
        print(kurt_price, skew_price, kurt_pct, skew_pct)
        # 4. Print the results
        print(f"\n--- Intersection of all criteria ---")
        print(f"Found {len(intersection)} stocks.")
        print(intersection)

    else:
        print("No data loaded from the database.")

    # 5. Close the database connection
    sql_op.close()