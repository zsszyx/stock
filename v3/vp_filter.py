import pandas as pd
from vp import calculate_volume_profile
from volume_analyzer import calculate_band_volumes
from plot_mountain import plot_volume_mountain

if __name__ == '__main__':
    start_date = '2025-11-12'
    end_date = '2025-12-09'
    
    vp_table_list = calculate_volume_profile(start_date, end_date)
    
    if not vp_table_list:
        print("No volume profile data found.")
    else:
        all_vp_df = pd.concat(vp_table_list, ignore_index=True)

        # Group by stock code and analyze each one
        for code, group_df in all_vp_df.groupby('code'):
            print(f"Analyzing stock: {code}")
            
            band_volumes, latest_price, price_band_width = calculate_band_volumes(group_df)
            
            if band_volumes.empty:
                print(f"No band volume data for stock {code}, skipping.")
                continue

            # Find the current price band
            current_band_center = (latest_price / price_band_width).round() * price_band_width
            
            # Get the volumes for the current band and its neighbors
            target_bands = [
                current_band_center - price_band_width,
                current_band_center,
                current_band_center + price_band_width
            ]
            
            target_volume = band_volumes[band_volumes['price_band_center'].isin(target_bands)]['volume'].sum()
            total_volume = band_volumes['volume'].sum()
            
            if total_volume > 0:
                volume_ratio = target_volume / total_volume
                
                if volume_ratio > 0.3:
                    print(f"  -> Stock {code} meets the criteria!")
                    print(f"     - Target Volume (3 bands): {target_volume:,.0f}")
                    print(f"     - Total Volume: {total_volume:,.0f}")
                    print(f"     - Ratio: {volume_ratio:.2%}")
                    
                    # Plot the volume mountain for the qualifying stock
                    plot_volume_mountain(group_df, code)
                    print(f"     - Plotted volume mountain for {code}.")

        print("Analysis and plotting complete.")