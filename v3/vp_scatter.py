import pandas as pd
from vp import calculate_volume_profile
from plot_scatter import plot_volume_scatter

if __name__ == '__main__':
    start_date = '2025-11-12'
    end_date = '2025-11-26'
    
    vp_table_list = calculate_volume_profile(start_date, end_date, freq='minute5')
    
    if not vp_table_list:
        print("No volume profile data found.")
    else:
        all_vp_df = pd.concat(vp_table_list, ignore_index=True)

        # Group by stock code and plot each one
        for code, group_df in all_vp_df.groupby('code'):
            print(f"Processing stock: {code}")
            plot_volume_scatter(group_df, code)

        print("Scatter plots saved in the 'plots_scatter' directory.")