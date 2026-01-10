import sys
import os
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sql_op.op import SqlOp
from scipy.stats import entropy
from strategy.period import add_week_index, add_reverse_month_index

import numpy as np


def _calculate_weighted_metrics(group):
    """Helper function to calculate weighted metrics for a group."""
    real_price = group['real_price']
    volume = group['volume'].astype(float)

    # Avoid division by zero if total volume is zero
    if volume.sum() == 0:
        raise ValueError("Total volume is zero, cannot calculate weighted metrics.")

    # Calculate weighted mean
    weighted_mean = np.average(real_price, weights=volume)

    # Calculate weighted variance and standard deviation
    weighted_variance = np.average((real_price - weighted_mean)**2, weights=volume)
    weighted_std = np.sqrt(weighted_variance)

    # Calculate weighted skewness
    if weighted_std == 0:
        weighted_skew = 0
    else:
        third_moment = np.average((real_price - weighted_mean)**3, weights=volume)
        weighted_skew = third_moment / (weighted_std**3)

    # Calculate weighted kurtosis (excess kurtosis)
    if weighted_std == 0:
        weighted_kurtosis = 0
    else:
        fourth_moment = np.average((real_price - weighted_mean)**4, weights=volume)
        # Subtract 3 for excess kurtosis, consistent with pandas' default
        weighted_kurtosis = (fourth_moment / (weighted_std**4)) - 3

    return pd.Series({
        'weighted_mean': weighted_mean,
        'weighted_std': weighted_std,
        'weighted_skew': weighted_skew,
        'weighted_kurtosis': weighted_kurtosis
    })


def calculate_distribution_metrics(df):
    """
    Calculates weighted distribution metrics for the 'real_price' for each stock,
    grouped by code and month.

    It computes the weighted mean, standard deviation, skewness, and kurtosis
    of the 'real_price', using 'volume' as the weight.

    Args:
        df (pd.DataFrame): Input DataFrame containing stock data with 'code',
                           'time', 'amount', and 'volume' columns.

    Returns:
        pd.DataFrame: A DataFrame with columns for 'code', 'month_index',
                      'weighted_mean', 'weighted_std', 'weighted_skew',
                      and 'weighted_kurtosis'.
    """
    df.loc[:, 'volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df.loc[:, 'amount'] = pd.to_numeric(df['amount'], errors='coerce')

    # Drop rows where 'volume' or 'amount' are NaN or volume is 0
    df.dropna(subset=['volume', 'amount'], inplace=True)
    df = df[df['volume'] > 0].copy()

    if df.empty:
        return pd.DataFrame()

    # Calculate real_price
    df.loc[:, 'real_price'] = df['amount'] / df['volume']

    # Add reverse month index for grouping
    df = add_reverse_month_index(df)

    # Group by stock code and month index, then apply the weighted calculation
    distribution_metrics = df.groupby(['code', 'week_index']).apply(_calculate_weighted_metrics, include_groups=False).reset_index()

    return distribution_metrics


def _calculate_weighted_pct_change_metrics(group):
    """Helper function to calculate weighted metrics for pct_change."""
    if group.empty or group['volume'].sum() == 0:
        return pd.Series({
            'weighted_mean_pct_chg': np.nan, 'weighted_std_pct_chg': np.nan,
            'weighted_skew_pct_chg': np.nan, 'weighted_kurt_pct_chg': np.nan,
            'weighted_mean_abs_pct_chg': np.nan, 'weighted_std_abs_pct_chg': np.nan,
            'weighted_skew_abs_pct_chg': np.nan, 'weighted_kurt_abs_pct_chg': np.nan
        })

    volume = group['volume'].astype(float)
    
    # Metrics for pct_chg
    pct_chg = group['pct_chg']
    weighted_mean_pct_chg = np.average(pct_chg, weights=volume)
    weighted_var_pct_chg = np.average((pct_chg - weighted_mean_pct_chg)**2, weights=volume)
    weighted_std_pct_chg = np.sqrt(weighted_var_pct_chg)
    
    if weighted_std_pct_chg == 0:
        weighted_skew_pct_chg = 0
        weighted_kurt_pct_chg = 0
    else:
        third_moment_pct_chg = np.average((pct_chg - weighted_mean_pct_chg)**3, weights=volume)
        weighted_skew_pct_chg = third_moment_pct_chg / (weighted_std_pct_chg**3)
        fourth_moment_pct_chg = np.average((pct_chg - weighted_mean_pct_chg)**4, weights=volume)
        weighted_kurt_pct_chg = (fourth_moment_pct_chg / (weighted_std_pct_chg**4)) - 3

    # Metrics for abs_pct_chg
    abs_pct_chg = group['abs_pct_chg']
    weighted_mean_abs_pct_chg = np.average(abs_pct_chg, weights=volume)
    weighted_var_abs_pct_chg = np.average((abs_pct_chg - weighted_mean_abs_pct_chg)**2, weights=volume)
    weighted_std_abs_pct_chg = np.sqrt(weighted_var_abs_pct_chg)

    if weighted_std_abs_pct_chg == 0:
        weighted_skew_abs_pct_chg = 0
        weighted_kurt_abs_pct_chg = 0
    else:
        third_moment_abs_pct_chg = np.average((abs_pct_chg - weighted_mean_abs_pct_chg)**3, weights=volume)
        weighted_skew_abs_pct_chg = third_moment_abs_pct_chg / (weighted_std_abs_pct_chg**3)
        fourth_moment_abs_pct_chg = np.average((abs_pct_chg - weighted_mean_abs_pct_chg)**4, weights=volume)
        weighted_kurt_abs_pct_chg = (fourth_moment_abs_pct_chg / (weighted_std_abs_pct_chg**4)) - 3

    return pd.Series({
        'weighted_mean_pct_chg': weighted_mean_pct_chg,
        'weighted_std_pct_chg': weighted_std_pct_chg,
        'weighted_skew_pct_chg': weighted_skew_pct_chg,
        'weighted_kurt_pct_chg': weighted_kurt_pct_chg,
        'weighted_mean_abs_pct_chg': weighted_mean_abs_pct_chg,
        'weighted_std_abs_pct_chg': weighted_std_abs_pct_chg,
        'weighted_skew_abs_pct_chg': weighted_skew_abs_pct_chg,
        'weighted_kurt_abs_pct_chg': weighted_kurt_abs_pct_chg
    })


def calculate_pct_change_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates distribution metrics for the daily percentage change ('pct_chg')
    of stock prices, grouped by code and month.

    It computes the skew, kurtosis, mean, and standard deviation for both
    'pct_chg' and its absolute value.

    Args:
        df (pd.DataFrame): Input DataFrame with 'code', 'date', and 'close' columns.

    Returns:
        pd.DataFrame: A DataFrame with metrics for 'pct_chg' and 'abs_pct_chg'.
    """
    if not all(col in df.columns for col in ['code', 'date', 'close', 'volume']):
        raise ValueError("Input DataFrame must have 'code', 'date', 'close', and 'volume' columns.")

    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['code', 'date'])
    
    df.loc[:, 'volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df.loc[:, 'close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(subset=['volume', 'close'], inplace=True)

    # Calculate daily percentage change
    df.loc[:, 'pct_chg'] = df.groupby('code')['close'].pct_change()

    # Calculate the absolute value of the percentage change
    df.loc[:, 'abs_pct_chg'] = df['pct_chg'].abs()
    
    df.dropna(subset=['pct_chg'], inplace=True)

    # Add reverse month index for grouping
    df = add_reverse_month_index(df)

    # Group by code and month, then apply the weighted calculation
    result_df = df.groupby(['code', 'week_index']).apply(_calculate_weighted_pct_change_metrics, include_groups=False).reset_index()

    return result_df


def _calculate_divergence_for_code(df_code, window_size=5, step=1, num_bins=50):
    """Helper to calculate rolling KL divergence for a single stock code based on trading days."""
    df_code = df_code.sort_values('date').reset_index(drop=True)
    
    # Get unique dates for this stock
    unique_dates = df_code['date'].dt.date
    unique_dates = pd.Series(unique_dates).unique()
    unique_dates.sort()

    results = []

    # Iterate through unique dates, ensuring we have enough history to form two windows
    for i in range(window_size, len(unique_dates)):
        
        # Define the date ranges for the two consecutive windows
        # Window P (current): [t-window_size+1, t]
        date_end_p = unique_dates[i]
        date_start_p = unique_dates[i - window_size + 1]
        
        # Window Q (previous): [t-window_size, t-1]
        date_end_q = unique_dates[i - 1]
        date_start_q = unique_dates[i - window_size]

        # Select data for the windows
        window_p = df_code[(df_code['date'].dt.date >= date_start_p) & (df_code['date'].dt.date <= date_end_p)]
        window_q = df_code[(df_code['date'].dt.date >= date_start_q) & (df_code['date'].dt.date <= date_end_q)]

        if window_p.empty or window_q.empty or window_p['volume'].sum() == 0 or window_q['volume'].sum() == 0:
            continue

        # Determine common price bins for both distributions
        min_price = min(window_p['real_price'].min(), window_q['real_price'].min())
        max_price = max(window_p['real_price'].max(), window_q['real_price'].max())

        if min_price == max_price:
            kl_div = 0.0
        else:
            bins = np.linspace(min_price, max_price, num_bins + 1)
            
            # Explicitly convert to float arrays to prevent dtype errors
            real_price_p = window_p['real_price'].to_numpy(dtype=float)
            volume_p = window_p['volume'].to_numpy(dtype=float)
            
            real_price_q = window_q['real_price'].to_numpy(dtype=float)
            volume_q = window_q['volume'].to_numpy(dtype=float)

            # Create weighted histograms
            hist_p, _ = np.histogram(real_price_p, bins=bins, weights=volume_p)
            hist_q, _ = np.histogram(real_price_q, bins=bins, weights=volume_q)

            # Normalize to get probability distributions, handle sum is 0 case
            sum_hist_p = hist_p.sum()
            sum_hist_q = hist_q.sum()

            if sum_hist_p == 0 or sum_hist_q == 0:
                continue

            dist_p = hist_p / sum_hist_p
            dist_q = hist_q / sum_hist_q
            
            # Calculate KL Divergence (P || Q)
            epsilon = 1e-10
            kl_div = entropy(dist_p + epsilon, dist_q + epsilon)

        results.append({
            'date': pd.to_datetime(date_end_p), # Use the end date of the current window
            'kl_divergence': kl_div
        })

    return pd.DataFrame(results)


def calculate_rolling_distribution_divergence(df, window_size=5, step=1):
    """
    Calculates the rolling KL divergence of the weighted real price distribution.

    For each stock, it computes the KL divergence between the distribution of
    days [t-5, t] and [t-6, t-1] with a step of 5 days.

    Args:
        df (pd.DataFrame): DataFrame with daily stock data, including 'code', 'date',
                           'amount', and 'volume'.
        window_size (int): The size of the rolling window (e.g., 5 days).
        step (int): The step size for the rolling calculation.

    Returns:
        pd.DataFrame: A DataFrame with 'code', 'date', and 'kl_divergence'.
    """
    if not all(col in df.columns for col in ['code', 'date', 'amount', 'volume']):
        raise ValueError("Input DataFrame must have 'code', 'date', 'amount', and 'volume' columns.")

    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    df.loc[:, 'amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df.loc[:, 'volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df.dropna(subset=['amount', 'volume'], inplace=True)
    df = df[df['volume'] > 0].copy()

    if df.empty:
        return pd.DataFrame()

    df.loc[:, 'real_price'] = df['amount'] / df['volume']

    # Group by code and apply the divergence calculation
    divergence_df = df.groupby('code').apply(
        _calculate_divergence_for_code,
        window_size=window_size,
        step=step
    ).reset_index()

    # Drop the extra 'level_1' column that may be created by apply
    if 'level_1' in divergence_df.columns:
        divergence_df.drop(columns=['level_1'], inplace=True)

    return divergence_df


if __name__ == '__main__':
    from sql_op import sql_config
    from sql_op.op import SqlOp

    sql_op = SqlOp()
    # Load daily k-data for calculating rolling divergence
    k_data = sql_op.read_k_data_by_date_range(
        sql_config.mintues5_table_name,
        start_date='2025-12-29',
        end_date='2026-01-10'
    )

    if k_data is not None and not k_data.empty:
        print("Calculating rolling distribution divergence...")
        divergence_results = calculate_rolling_distribution_divergence(k_data)
        
        if not divergence_results.empty:
            print("\nRolling Divergence Results:")
            # Sort by divergence to see the most significant shifts
            print(divergence_results.sort_values('kl_divergence', ascending=False).head(20))
        else:
            print("Could not calculate rolling divergence.")
    else:
        print("No data to process for the selected date range.")

    sql_op.close()