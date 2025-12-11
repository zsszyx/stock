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
              containing 'volatility', 'volume_profile', and 'focused_volume_ratio'.
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

        # Filter out stocks with zero volatility
        if price_volatility == 0:
            continue
        
        # Get latest price
        latest_price = stock_df['price'].iloc[-1]
        
        # Round the price to 2 decimal places
        stock_df['price'] = stock_df['price'].round(2)

        # Group by time and price, then sum the volume
        volume_profile = stock_df.groupby(['time', 'price'])['volume'].sum().reset_index()
        volume_profile = volume_profile.rename(columns={'volume': 'total_volume'})
        
        # Calculate focused volume ratio
        band_width = price_volatility
        focused_volume_ratio = 0
        if band_width > 0:
            total_stock_volume = volume_profile['total_volume'].sum()
            if total_stock_volume > 0:
                latest_price_band_index = round(latest_price / band_width)
                target_band_indices = [latest_price_band_index - 1, latest_price_band_index, latest_price_band_index + 1]
                
                volume_profile['band_index'] = (volume_profile['price'] / band_width).round()
                focused_volume = volume_profile[volume_profile['band_index'].isin(target_band_indices)]['total_volume'].sum()
                focused_volume_ratio = focused_volume / total_stock_volume

        stock_attributes[code] = {
            'volatility': price_volatility,
            'volume_profile': volume_profile,
            'focused_volume_ratio': focused_volume_ratio
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
    
    if not stock_attributes:
        print("No stock data found.")
    else:
        # Extract volatility and print
        volatility_dict = {code: data['volatility'] for code, data in stock_attributes.items()}
        print("Price Volatility by Stock Code:")
        print(volatility_dict)
        if volatility_dict:
            mean_volatility = np.mean(list(volatility_dict.values()))
            print(f"Mean volatility of all stocks: {mean_volatility}")

        # Extract volume profiles and concatenate
        vp_table_list = [data['volume_profile'] for data in stock_attributes.values() if 'volume_profile' in data]
        
        if not vp_table_list:
            print("No volume profile data found.")
        else:
            all_vp_df = pd.concat(vp_table_list, ignore_index=True)
            print(f"save all volume profile to volume_profiles.csv")
            all_vp_df.to_csv("volume_profiles.csv", index=False, encoding='utf-8-sig')

            # Group by stock code
            for code, group_df in all_vp_df.groupby('code'):
                group_df['date'] = pd.to_datetime(group_df['date'])
                unique_dates = sorted(group_df['date'].unique())
                
                # Ensure there are enough dates to split into 4 periods
                if len(unique_dates) < 4:
                    continue

                # Split dates into 4 periods
                n_dates = len(unique_dates)
                period_splits = np.array_split(unique_dates, 4)
                
                periods_data = []
                valid_periods = True
                for i, period_dates in enumerate(period_splits):
                    period_df = group_df[group_df['date'].isin(period_dates)]
                    
                    # Aggregate volume by price for the period
                    period_vp = period_df.groupby('price')['total_volume'].sum().reset_index()
                    period_vp.rename(columns={'total_volume': 'total_volume'}, inplace=True)

                    if period_vp.empty:
                        valid_periods = False
                        break

                    poc, va_high, va_low = calculate_value_area_and_poc(period_vp)
                    
                    if poc is None or va_high is None or va_low is None:
                        valid_periods = False
                        break
                    
                    # Calculate summed volume in VA range
                    va_volume = period_vp[(period_vp['price'] >= va_low) & (period_vp['price'] <= va_high)]['total_volume'].sum()
                    
                    periods_data.append({
                        'poc': poc,
                        'va_high': va_high,
                        'va_low': va_low,
                        'va_volume': va_volume
                    })

                if not valid_periods or len(periods_data) != 4:
                    continue

                # New filtering logic
                is_period1_ok = periods_data[0]['va_volume'] == max(p['va_volume'] for p in periods_data)

                is_period4_ok = (periods_data[3]['poc'] >= periods_data[2]['poc']) and (periods_data[3]['va_volume'] <= 0.5 * periods_data[0]['va_volume'])

                is_period2_ok = periods_data[1]['poc'] == max(p['poc'] for p in periods_data)

                is_period3_ok = periods_data[2]['poc'] == min(p['poc'] for p in periods_data) and (periods_data[2]['va_volume'] == min(p['va_volume'] for p in periods_data)) 


                if is_period1_ok and is_period2_ok and is_period3_ok and is_period4_ok:
                    print(f"Stock {code} matches the criteria. Plotting...")
                    plot_volume_scatter(group_df, code, period_splits)