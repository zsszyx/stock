import sys
import os
import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sql_op.op import SqlOp
from sql_op import sql_config

def add_week_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a reverse chronological week index to the DataFrame based on a 7-day period.

    The index starts from 0 for the most recent 7-day period and increases for
    earlier periods. This ensures the most recent data chunk is always a full week.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'time' column.

    Returns:
        pd.DataFrame: The DataFrame with an added 'week_index' column.
    """
    if 'time' not in df.columns or df.empty:
        return df

    # Ensure 'time' column is in datetime format
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M%S%f', errors='coerce')
    df.dropna(subset=['time'], inplace=True)

    if df.empty:
        return df

    # Find the most recent date in the dataset
    max_date = df['time'].max()

    # Calculate the number of days from the max_date for each entry
    days_from_max = (max_date - df['time']).dt.days

    # Calculate the reverse week index
    df['week_index'] = days_from_max // 7
    
    return df

def add_reverse_month_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a reverse chronological month index to the DataFrame based on a 30-day period.

    The index starts from 0 for the most recent 30-day period and increases for
    earlier periods. This ensures the most recent data chunk is always a full month.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'time' column.

    Returns:
        pd.DataFrame: The DataFrame with an added 'month_index' column.
    """
    if 'time' not in df.columns or df.empty:
        return df

    # Ensure 'time' column is in datetime format
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M%S%f', errors='coerce')
    df.dropna(subset=['time'], inplace=True)

    if df.empty:
        return df

    # Find the most recent date in the dataset
    max_date = df['time'].max()

    # Calculate the number of days from the max_date for each entry
    days_from_max = (max_date - df['time']).dt.days

    # Calculate the reverse month index
    df['month_index'] = days_from_max // 30
    
    return df

if __name__ == '__main__':
    # Example usage:
    sql_op = SqlOp()
    # Load some data from the database for testing
    k_data = sql_op.read_k_data_by_date_range(sql_config.mintues5_table_name, '2025-10-01', '2026-01-01')

    if k_data is not None and not k_data.empty:
        # --- Test Week Index ---
        print("--- Testing Week Index ---")
        k_data_with_week = add_week_index(k_data.copy())
        print("Week index values:")
        print(k_data_with_week['week_index'].value_counts().sort_index())
        print("\n")

        # --- Test Reverse Month Index ---
        print("--- Testing Reverse Month Index ---")
        k_data_with_month = add_reverse_month_index(k_data.copy())
        print("Reverse month index values:")
        print(k_data_with_month['month_index'].value_counts().sort_index())
        print("Sample data with month index:")
        print(k_data_with_month[['time', 'month_index']].tail())
    else:
        print("No data to process.")

    sql_op.close()