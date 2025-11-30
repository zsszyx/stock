import pandas as pd

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

    va_high = max(va_prices) if va_prices else None
    va_low = min(va_prices) if va_prices else None

    return poc_price, va_high, va_low