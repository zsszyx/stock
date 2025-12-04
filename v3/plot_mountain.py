import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_volume_mountain(vp_df, stock_code):
    """
    Plots a volume mountain chart for a given stock.

    Args:
        vp_df (pd.DataFrame): DataFrame with 'date', 'price', and 'total_volume'.
        stock_code (str): The stock code for labeling the plot.
    """
    if vp_df.empty:
        print(f"No data for stock {stock_code}, skipping plot.")
        return

    # Ensure date is datetime and sort
    vp_df['date'] = pd.to_datetime(vp_df['date'])
    vp_df = vp_df.sort_values(by='date')

    unique_dates = vp_df['date'].unique()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use a colormap to distinguish different days
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_dates)))

    # Initialize a dataframe to store cumulative volumes
    cumulative_vp = pd.DataFrame()

    for i, date in enumerate(unique_dates):
        daily_vp = vp_df[vp_df['date'] == date].groupby('price')['total_volume'].sum().reset_index()
        
        if i == 0:
            cumulative_vp = daily_vp
        else:
            # Merge with previous cumulative data
            merged = pd.merge(cumulative_vp, daily_vp, on='price', how='outer', suffixes=['_cum', '_new'])
            merged.fillna(0, inplace=True)
            merged['total_volume'] = merged['total_volume_cum'] + merged['total_volume_new']
            cumulative_vp = merged[['price', 'total_volume']]

        # Sort by price for a clean plot
        cumulative_vp = cumulative_vp.sort_values(by='price')

        # Plot the cumulative volume for the day
        ax.plot(cumulative_vp['total_volume'], cumulative_vp['price'], color=colors[i], label=pd.to_datetime(date).strftime('%Y-%m-%d'))

    # Formatting the plot
    ax.set_title(f'Volume Mountain Map for {stock_code}')
    ax.set_xlabel('Cumulative Volume')
    ax.set_ylabel('Price')
    ax.legend(title="Date", loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Ensure the plot directory exists
    plots_dir = 'd:\\stock\\plots_mountain'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Save the plot
    plt.savefig(os.path.join(plots_dir, f'mountain_profile_{stock_code}.png'))
    plt.close(fig)