import pandas as pd

def calculate_band_volumes(vp_df):
    """
    Calculates the total volume for each price band.

    Args:
        vp_df (pd.DataFrame): DataFrame with 'time', 'price', and 'total_volume'.

    Returns:
        pd.DataFrame: A DataFrame with 'price_band_center' and 'volume'.
    """
    if vp_df.empty:
        return pd.DataFrame(columns=['price_band_center', 'volume'])

    # Pivot to get prices on rows, time on columns
    pivot_df = vp_df.pivot_table(index='price', columns='time', values='total_volume', fill_value=0)
    
    # Calculate cumulative sum across time (columns)
    cumulative_pivot = pivot_df.cumsum(axis=1)
    
    # Get the latest price to use for clustering and reference
    latest_price = vp_df.sort_values(by='time', ascending=True).iloc[-1]['price']

    # --- Price Band Clustering ---
    price_band_width = latest_price * 0.005
    final_cumulative_volume = cumulative_pivot.iloc[:, -1]
    
    # Create a dataframe from the final volumes
    final_volume_df = final_cumulative_volume.reset_index()
    final_volume_df.columns = ['price', 'volume']
    
    # Assign each price to a price band
    final_volume_df['price_band_center'] = (final_volume_df['price'] / price_band_width).round() * price_band_width
    
    # Group by the price band and sum the volumes
    clustered_data = final_volume_df.groupby('price_band_center')['volume'].sum().reset_index()
    
    return clustered_data, latest_price, price_band_width