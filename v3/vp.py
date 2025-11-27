import pandas as pd
from infra import get_stock_merge_table

def calculate_volume_profile(start_date, end_date, freq='minute5'):
    """
    Calculates the volume profile for each stock.

    Args:
        start_date (str): The start date for the data.
        end_date (str): The end date for the data.
        freq (str, optional): The frequency of the data. Defaults to 'minute5'.

    Returns:
        list: A list of pandas DataFrames, where each DataFrame is the volume profile for a stock.
    """
    # Get the merged stock data table
    df = get_stock_merge_table(freq=freq, start_date=start_date, end_date=end_date)

    # Reset index to make 'code' and 'date' columns
    df = df.reset_index()

    # Group by stock code
    grouped_by_code = df.groupby('code')

    # Calculate volume profile for each stock
    vp_table_list = []
    for code, stock_df in grouped_by_code:
        # Group by price and sum the volume
        volume_profile = stock_df.groupby('price')['volume'].sum().reset_index()
        volume_profile = volume_profile.rename(columns={'volume': 'total_volume'})
        volume_profile['code'] = code
        vp_table_list.append(volume_profile)

    return vp_table_list

def calculate_value_area_and_poc(vp_df):
    """
    Calculates the Point of Control (POC) and Value Area (VA) for a given volume profile.

    Args:
        vp_df (pd.DataFrame): A DataFrame with 'price' and 'total_volume' columns.

    Returns:
        tuple: A tuple containing POC, VA High, and VA Low.
    """
    if vp_df.empty:
        return None, None, None

    # Find the Point of Control (POC)
    poc_index = vp_df['total_volume'].idxmax()
    poc_price = vp_df.loc[poc_index, 'price']

    total_volume = vp_df['total_volume'].sum()
    target_volume = total_volume * 0.7

    # Sort by volume to find the most significant price levels
    vp_sorted_by_volume = vp_df.sort_values(by='total_volume', ascending=False).reset_index(drop=True)

    # Accumulate volume until 70% is reached
    cumulative_volume = 0
    va_prices = []
    for index, row in vp_sorted_by_volume.iterrows():
        cumulative_volume += row['total_volume']
        va_prices.append(row['price'])
        if cumulative_volume >= target_volume:
            break

    va_high = max(va_prices)
    va_low = min(va_prices)

    return poc_price, va_high, va_low

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

if __name__ == "__main__":
    start_date = '2025-11-12'
    end_date = '2025-11-26'
    
    # Calculate the volume profiles
    vptables = calculate_volume_profile(start_date=start_date, end_date=end_date)
    
    # This dictionary will store basic attributes for each stock.
    stock_attributes = {}

    # Print the volume profile for the first stock as an example
    if vptables:
        print("Volume profile for the first stock:")
        print(vptables[0])
        
        # You can also concatenate all profiles into a single DataFrame
        all_vps = pd.concat(vptables, ignore_index=True)
        print("\nAll volume profiles combined:")
        print(all_vps)
        all_vps.to_csv('volume_profiles.csv', index=False)
        print("\nSaved all volume profiles to volume_profiles.csv")

        # Calculate and store VA and POC for each stock
        for vp_df in vptables:
            if not vp_df.empty:
                code = vp_df['code'].iloc[0]
                poc, va_high, va_low = calculate_value_area_and_poc(vp_df)
                hvns, lvns = find_hvn_lvn(vp_df)
                stock_attributes[code] = {
                    'va_high': va_high,
                    'va_low': va_low,
                    'poc': poc,
                    'hvns': hvns,
                    'lvns': lvns
                }
        
        print("\nStock Attributes (VA, POC, HVNs, and LVNs):")
        print(stock_attributes)