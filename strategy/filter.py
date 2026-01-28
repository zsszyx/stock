import sys
import os
import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sql_op.op import SqlOp
from sql_op import sql_config

def apply_length_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a length filter to the DataFrame.

    It calculates the number of entries for each stock code, finds the maximum
    length, and adds a 'length_filter' column. This column is True for stocks
    whose length is at least 90% of the maximum length, and False otherwise.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'code' column.

    Returns:
        pd.DataFrame: The DataFrame with the added 'length_filter' column.
    """
    if 'code' not in df.columns:
        raise ValueError("Input DataFrame must have a 'code' column.")

    # Calculate the length for each stock code
    code_lengths = df.groupby('code').size()
    # print(f'code_lengths: {code_lengths}')

    if code_lengths.empty:
        df['length_filter'] = False
        return df

    # Find the maximum length
    mean_length = code_lengths.mean()

    # Determine the threshold
    length_threshold = mean_length 

    # Identify the codes that pass the filter
    passing_codes = code_lengths[code_lengths >= length_threshold].index
    print(f'passing_codes: {len(passing_codes)}')

    # Add the 'length_filter' column
    df['length_filter'] = df['code'].isin(passing_codes)

    return df

def apply_nan_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters data based on the count of non-NaN rows for each stock code.

    A stock is kept if its count of non-NaN rows is greater than or equal to the
    average count of non-NaN rows across all stocks.

    Args:
        df (pd.DataFrame): The input DataFrame with stock data, must contain a 'code' column.

    Returns:
        pd.DataFrame: The DataFrame with an added 'nan_filter' boolean column.
    """
    # Drop rows with any NaN values and count the remaining rows for each stock
    non_nan_lengths = df.dropna().groupby('code').size()

    # Calculate the average length of non-NaN data
    if non_nan_lengths.empty:
        # If there's no data left after dropping NaNs, filter everything out
        df['nan_filter'] = False
        return df
        
    average_length = non_nan_lengths.mean()
    print(f"Average non-NaN data length: {average_length}")

    # Identify which codes meet the length requirement
    codes_to_keep = non_nan_lengths[non_nan_lengths >= average_length].index
    print(f"Codes to keep: {len(codes_to_keep)}")

    # Add the 'nan_filter' column
    df['nan_filter'] = df['code'].isin(codes_to_keep)
    
    return df

def apply_increase_filter(df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """
    Filters stocks based on their maximum price increase over a specified period.

    For each stock, this function calculates the maximum percentage increase from
    the lowest price to the highest price within the last `days` days. If the
    maximum increase is more than 8%, the stock is filtered out (marked as False).

    Args:
        df (pd.DataFrame): DataFrame with stock data, including 'code', 'date',
                           'low', and 'high' columns.
        days (int): The number of days to look back for calculating the increase.

    Returns:
        pd.DataFrame: The DataFrame with an added 'increase_filter' boolean column.
    """
    if not all(col in df.columns for col in ['code', 'date', 'low', 'high']):
        raise ValueError("DataFrame must include 'code', 'date', 'low', and 'high' columns.")

    df['date'] = pd.to_datetime(df['date'])
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    end_date = df['date'].max()
    start_date = end_date - pd.Timedelta(days=days)

    # Filter the DataFrame to the relevant date range
    recent_data = df[df['date'] >= start_date]

    # Find the min 'low' and max 'high' for each stock in the period
    min_low = recent_data.groupby('code')['low'].min()
    max_high = recent_data.groupby('code')['high'].max()

    # Calculate the percentage increase
    increase = (max_high - min_low) / min_low

    # Identify codes with an increase of more than 8%
    filtered_codes = increase[increase > 0.08].index

    # Mark all data for these codes as False
    df['increase_filter'] = ~df['code'].isin(filtered_codes)

    return df

def apply_volume_zero_filter(df: pd.DataFrame) -> pd.DataFrame:
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['volume_filter'] = df['volume'] > 0
    return df

if __name__ == '__main__':
    # This is a sample usage of the functions in this file
    import numpy as np

    sql_op = SqlOp()
    
    # Example for apply_length_filter
    print("--- Testing Length Filter ---")
    k_data_length = sql_op.read_k_data_by_date_range(sql_config.mintues5_table_name, '2025-12-01', '2026-01-01')
    if k_data_length is not None and not k_data_length.empty:
        filtered_data_length = apply_length_filter(k_data_length)
        filtered_data_length = apply_nan_filter(filtered_data_length)
        filtered_data_length = apply_increase_filter(filtered_data_length)


    sql_op.close()