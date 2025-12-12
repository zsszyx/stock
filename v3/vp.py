import pandas as pd
import numpy as np
from infra import get_stock_merge_table
from plot import plot_volume_scatter
from utils import calculate_value_area_and_poc

def calculate_volume_profile(start_date, end_date, freq='minute5'):
    """
    Calculates the volume profile and other attributes for each stock.

    Args:
        start_date (str): The start date for the data.
        end_date (str): The end date for the data.
        freq (str, optional): The frequency of the data. Defaults to 'minute5'.

    Returns:
        dict: A dictionary where keys are stock codes and values are dictionaries
              containing 'volatility', 'volume_profile', 'focused_volume_ratio',
              and 'recent_focused_concentration_ratio'.
    """
    # Get the merged stock data table
    df = get_stock_merge_table(freq=freq, start_date=start_date, end_date=end_date)

    # Reset index to make 'code', 'time', and 'price' columns
    df = df.reset_index()

    # Group by stock code
    grouped_by_code = df.groupby('code')

    # Calculate volume profile and other attributes for each stock
    stock_attributes = {}
    for code, stock_df in grouped_by_code:
        # Convert price to numeric, coercing errors to NaN
        stock_df['price'] = pd.to_numeric(stock_df['price'], errors='coerce')
        stock_df.dropna(subset=['price'], inplace=True)

        # Calculate price volatility
        price_volatility = stock_df['price'].std()
        
        # Get latest price
        latest_price = stock_df['price'].iloc[-1]
        
        # Round the price to 2 decimal places
        # stock_df['price'] = stock_df['price'].round(2)

        # Group by time and price, then sum the volume
        volume_profile = stock_df.groupby(['time', 'price'])['volume'].reset_index()
        volume_profile = volume_profile.rename(columns={'volume': 'total_volume'})
        
        # Calculate focused volume ratios
        band_width = price_volatility
        focused_volume_ratio = 0
        recent_focused_concentration_ratio = 0

        if band_width > 0:
            total_stock_volume = volume_profile['total_volume'].sum()
            if total_stock_volume > 0:
                # Determine target bands
                latest_price_band_index = round(latest_price / band_width)
                target_band_indices = [latest_price_band_index - 1, latest_price_band_index, latest_price_band_index + 1]
                
                # Add band index to volume_profile
                volume_profile['band_index'] = (volume_profile['price'] / band_width).round()
                
                # Filter for rows in the focused bands
                focused_bands_vp = volume_profile[volume_profile['band_index'].isin(target_band_indices)]
                
                # Calculate total volume in focused bands for the entire period
                total_focused_volume = focused_bands_vp['total_volume'].sum()

                # 1. Calculate focused_volume_ratio (focused volume / total stock volume)
                if total_stock_volume > 0:
                    focused_volume_ratio = total_focused_volume / total_stock_volume

                # 2. Calculate recent_focused_concentration_ratio (recent focused volume / total focused volume)
                unique_dates = sorted(volume_profile['time'].dt.date.unique())
                if len(unique_dates) >= 5:
                    last_5_days = unique_dates[-5:]
                    
                    # Filter the focused_bands_vp for the last 5 days
                    recent_focused_bands_vp = focused_bands_vp[focused_bands_vp['time'].dt.date.isin(last_5_days)]
                    
                    # Calculate volume in focused bands for the last 5 days
                    recent_focused_volume = recent_focused_bands_vp['total_volume'].sum()
                    
                    # Calculate the ratio
                    if total_focused_volume > 0:
                        recent_focused_concentration_ratio = recent_focused_volume / total_focused_volume

        stock_attributes[code] = {
            'volatility': price_volatility,
            'volume_profile': volume_profile,
            'focused_volume_ratio': focused_volume_ratio,
            'recent_focused_concentration_ratio': recent_focused_concentration_ratio
        }

    return stock_attributes

def find_hvn_lvn(vp_df, prominence=1.0):
    """
    Finds High Volume Nodes (HVNs) and Low Volume Nodes (LVNs) from a volume profile.

    Args:
        vp_df (pd.DataFrame): A DataFrame with 'price' and 'total_volume' columns.
        prominence (float): A factor to determine significance. HVNs are nodes with volume
                           greater than the mean + prominence * std, and LVNs are nodes with
                           volume less than the mean - prominence * std.

    Returns:
        tuple: A tuple containing a list of HVNs and a list of LVNs.
    """
    if vp_df.empty or len(vp_df) < 3:
        return [], []

    mean_volume = vp_df['total_volume'].mean()
    std_volume = vp_df['total_volume'].std()

    hvn_threshold = mean_volume + prominence * std_volume
    lvn_threshold = max(0, mean_volume - prominence * std_volume)  # Volume can't be negative

    hvns = vp_df[vp_df['total_volume'] > hvn_threshold]['price'].tolist()
    lvns = vp_df[vp_df['total_volume'] < lvn_threshold]['price'].tolist()

    return hvns, lvns

if __name__ == '__main__':
    start_date = '2025-11-12'
    end_date = '2025-12-04'
    
    stock_attributes = calculate_volume_profile(start_date, end_date)
       
    # New filtering logic based on stock_attributes
    if stock_attributes:
        # Sort by volatility and take the top half
        sorted_by_volatility = sorted(stock_attributes.items(), key=lambda item: item[1]['volatility'], reverse=True)
        top_half_volatility_codes = {code for code, data in sorted_by_volatility[:len(sorted_by_volatility) // 2]}

        # Filter the top half by focused_volume_ratio and take the top 100
        filtered_by_volatility = {code: data for code, data in stock_attributes.items() if code in top_half_volatility_codes}
        sorted_by_focused_volume = sorted(filtered_by_volatility.items(), key=lambda item: item[1]['focused_volume_ratio'], reverse=True)
        top_100_focused_volume_codes = {code for code, data in sorted_by_focused_volume[:100]}

        # Filter the result by recent_focused_concentration_ratio and take the top 100 smallest
        filtered_by_focused_volume = {code: data for code, data in stock_attributes.items() if code in top_100_focused_volume_codes}
        sorted_by_recent_concentration = sorted(filtered_by_focused_volume.items(), key=lambda item: item[1]['recent_focused_concentration_ratio'])
        top_100_recent_concentration_codes = {code for code, data in sorted_by_recent_concentration[:100]}
            
        # Final intersection of all filters
        final_selection = top_half_volatility_codes.intersection(top_100_focused_volume_codes).intersection(top_100_recent_concentration_codes)
            
        print("\nFiltered Stock Codes based on the new strategy:")
        print(final_selection)

