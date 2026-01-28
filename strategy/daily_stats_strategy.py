import pandas as pd
import numpy as np
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sql_op import op, sql_config
from strategy.filter import apply_volume_zero_filter

def calculate_weighted_stats(values, weights):
    """
    Computes weighted skewness and weighted kurtosis (Fisher, excess).
    """
    if len(values) < 2 or weights.sum() == 0:
        return 0.0, 0.0

    # Weighted Mean
    mean = np.average(values, weights=weights)
    
    # Weighted Variance
    variance = np.average((values - mean)**2, weights=weights)
    
    if variance == 0:
        return 0.0, 0.0

    std_dev = np.sqrt(variance)
    
    # Weighted Skewness
    # Formula: sum(w * (x - mean)^3) / (sum(w) * std^3)
    skew = np.average(((values - mean) / std_dev)**3, weights=weights)
    
    # Weighted Kurtosis (Excess)
    # Formula: sum(w * (x - mean)^4) / (sum(w) * std^4) - 3
    kurt = np.average(((values - mean) / std_dev)**4, weights=weights) - 3
    
    return skew, kurt

import pandas as pd
import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sql_op import op, sql_config
from strategy.filter import apply_volume_zero_filter

def calculate_weighted_stats(values, weights):
    """
    Computes weighted skewness and weighted kurtosis (Fisher, excess).
    """
    if len(values) < 2 or weights.sum() == 0:
        return 0.0, 0.0

    # Weighted Mean
    mean = np.average(values, weights=weights)
    
    # Weighted Variance
    variance = np.average((values - mean)**2, weights=weights)
    
    if variance == 0:
        return 0.0, 0.0

    std_dev = np.sqrt(variance)
    
    # Weighted Skewness
    skew = np.average(((values - mean) / std_dev)**3, weights=weights)
    
    # Weighted Kurtosis (Excess)
    kurt = np.average(((values - mean) / std_dev)**4, weights=weights) - 3
    
    return skew, kurt

def calculate_and_save_daily_stats(start_date: str, end_date: str):
    """
    Calculates daily stats for the given date range and saves to DB.
    """
    sql_operator = op.SqlOp()
    
    # 1. Read data
    k_data = sql_operator.read_k_data_by_date_range(sql_config.mintues5_table_name, start_date, end_date)
    
    if k_data.empty:
        print(f"No K-line data found for {start_date} to {end_date}.")
        return

    # Normalize data
    # Remove 'sh.'/'sz.' prefix from code
    k_data['code'] = k_data['code'].astype(str).apply(lambda x: x.split('.')[-1])
    
    # Convert numeric columns
    cols_to_numeric = ['amount', 'volume']
    for col in cols_to_numeric:
        k_data[col] = pd.to_numeric(k_data[col], errors='coerce')

    # Filter invalid data
    k_data = apply_volume_zero_filter(k_data)
    k_data = k_data[k_data['volume_filter']].copy()
    
    # Calculate 'price' for the bar (VWAP of the bar)
    k_data['bar_price'] = k_data['amount'] / k_data['volume']

    results = []

    # 2. Group by Date and Code
    # Grouping 5-min data by code/date.
    grouped = k_data.groupby(['date', 'code'])

    for (date, code), group in grouped:
        if len(group) < 3: # Need at least a few bars
            continue
            
        prices = group['bar_price']
        volumes = group['volume']
        
        # --- Stats ---
        # Unweighted (Pandas uses unbiased estimator)
        std_skew = prices.skew()
        std_kurt = prices.kurt()
        
        # Weighted
        w_skew, w_kurt = calculate_weighted_stats(prices.values, volumes.values)
        
        results.append({
            'date': date,
            'code': code,
            'skew': std_skew,
            'kurt': std_kurt,
            'weighted_skew': w_skew,
            'weighted_kurt': w_kurt,
            'volume_sum': volumes.sum()
        })

    if not results:
        print("No stats calculated.")
        return

    # 3. Create DataFrame and Save
    stats_df = pd.DataFrame(results).fillna(0)
    
    print(f"Saving {len(stats_df)} records to {sql_config.daily_stats_table_name}...")
    sql_operator.upsert_df_to_db(stats_df, sql_config.daily_stats_table_name, index=False)

if __name__ == '__main__':
    # Simple test run for a specific recent date
    calculate_and_save_daily_stats('2026-01-20', '2026-01-22')
