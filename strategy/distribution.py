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
    # Ensure 'volume' and 'amount' are numeric, coercing errors
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    # Drop rows where 'volume' or 'amount' are NaN or volume is 0
    df.dropna(subset=['volume', 'amount'], inplace=True)
    df = df[df['volume'] > 0]

    if df.empty:
        return pd.DataFrame()

    # Calculate real_price
    df['real_price'] = df['amount'] / df['volume']

    # Add reverse month index for grouping
    df = add_reverse_month_index(df)

    # Group by stock code and month index, then apply the weighted calculation
    distribution_metrics = df.groupby(['code', 'month_index']).apply(_calculate_weighted_metrics).reset_index()

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

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['code', 'date'])
    
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df.dropna(subset=['volume'], inplace=True)

    # Calculate daily percentage change
    df['pct_chg'] = df.groupby('code')['close'].pct_change()

    # Calculate the absolute value of the percentage change
    df['abs_pct_chg'] = df['pct_chg'].abs()
    
    df.dropna(subset=['pct_chg'], inplace=True)

    # Add reverse month index for grouping
    df = add_reverse_month_index(df)

    # Group by code and month, then apply the weighted calculation
    result_df = df.groupby(['code', 'month_index']).apply(_calculate_weighted_pct_change_metrics).reset_index()

    return result_df


def _calculate_kl_divergence_for_code(group, num_bins=50):
    """Helper function to calculate weekly KL divergence for a single stock code."""
    # Ensure data types are correct before processing
    group['real_price'] = pd.to_numeric(group['real_price'], errors='coerce')
    group['volume'] = pd.to_numeric(group['volume'], errors='coerce')
    group.dropna(subset=['real_price', 'volume'], inplace=True)

    # If after cleaning, the group is empty, return an empty DataFrame
    if group.empty:
        return pd.DataFrame(columns=['kl_divergence'])

    group = group.sort_values('week_index')
    
    # Determine global bins for this stock
    min_price = group['real_price'].min()
    max_price = group['real_price'].max()
    
    # If all prices are the same, divergence is not meaningful
    if min_price == max_price:
        return pd.DataFrame({
            'week_index': group['week_index'].unique(),
            'kl_divergence': np.nan
        }).set_index('week_index')

    bins = np.linspace(min_price, max_price, num_bins + 1)
    
    # Get weighted histogram for each week
    weekly_dists = {}
    for week, week_group in group.groupby('week_index'):
        # Use volume as weights for the histogram
        hist, _ = np.histogram(
            week_group['real_price'],
            bins=bins,
            weights=week_group['volume']
        )
        
        # Normalize to get a probability distribution
        if hist.sum() > 0:
            weekly_dists[week] = hist / hist.sum()
        else:
            weekly_dists[week] = np.zeros(num_bins)

    # Calculate KL divergence between consecutive weeks
    weeks = sorted(weekly_dists.keys())
    kl_divergences = {}
    
    # The first week has no previous week to compare to
    if weeks:
        kl_divergences[weeks[0]] = np.nan
        
    for i in range(1, len(weeks)):
        prev_week = weeks[i-1]
        curr_week = weeks[i]
        
        p_dist = weekly_dists[curr_week]
        q_dist = weekly_dists[prev_week]
        
        # Use scipy's entropy which calculates KL divergence: sum(p * log(p / q))
        # It handles p=0 and q=0 cases gracefully.
        # We add a small epsilon to avoid log(0) or division by zero if not handled.
        epsilon = 1e-10
        kl_div = entropy(p_dist + epsilon, q_dist + epsilon)
        kl_divergences[curr_week] = kl_div
        
    return pd.DataFrame.from_dict(kl_divergences, orient='index', columns=['kl_divergence'])


if __name__ == '__main__':
    from sql_op import sql_config
    from sql_op.op import SqlOp

    sql_op = SqlOp()
    # Load daily k-data for calculating pct_change metrics
    k_data = sql_op.read_k_data_by_date_range(
        sql_config.daily_table_name,
        start_date='2023-01-01',
        end_date='2024-01-01'
    )

    if k_data is not None and not k_data.empty:
        print("Calculating pct_change metrics...")
        pct_change_metrics_df = calculate_pct_change_metrics(k_data)
        
        if not pct_change_metrics_df.empty:
            print("\nPct Change Metrics Results:")
            print(pct_change_metrics_df.head(10))
        else:
            print("Could not calculate pct_change metrics.")
    else:
        print("No data to process for the selected date range.")

    sql_op.close()