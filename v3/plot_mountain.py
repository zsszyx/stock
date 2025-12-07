import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_volume_mountain(vp_df, stock_code):
    """
    Plots a volume mountain chart for a given stock.

    Args:
        vp_df (pd.DataFrame): DataFrame with 'time', 'price', and 'total_volume'.
        stock_code (str): The stock code for labeling the plot.
    """
    if vp_df.empty:
        print(f"No data for stock {stock_code}, skipping plot.")
        return

    vp_df['time'] = pd.to_datetime(vp_df['time'], format='%Y%m%d%H%M%S%f')
    
    # Pivot to get prices on rows, time on columns
    pivot_df = vp_df.pivot_table(index='price', columns='time', values='total_volume', fill_value=0)
    
    # Calculate cumulative sum across time (columns)
    cumulative_pivot = pivot_df.cumsum(axis=1)
    
    # Now plot each column (which represents a point in time)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    unique_times = cumulative_pivot.columns
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_times)))
    
    for i, time in enumerate(unique_times):
        # Get the data for the current time
        series = cumulative_pivot[time]
        # Filter out zero volumes to make plotting faster
        series = series[series > 0]
        if not series.empty:
            ax.plot(series.values, series.index, color=colors[i])

    # Get the final cumulative volume for each price
    final_cumulative_volume = cumulative_pivot.iloc[:, -1]

    # Determine the volume threshold for the top 70%
    top_70_percent_threshold = final_cumulative_volume.quantile(0.3)

    # Filter for prices with volume in the top 70%
    top_prices = final_cumulative_volume[final_cumulative_volume >= top_70_percent_threshold]

    # Plot markers and annotations for these top prices
    for price, volume in top_prices.items():
        ax.scatter(volume, price, color='blue', zorder=5)  # zorder to make sure markers are on top
        ax.text(volume, price, f' {volume:,.0f}', color='blue')

    # Get the latest price
    latest_price = vp_df.sort_values(by='time', ascending=True).iloc[-1]['price']

    # Add a horizontal line for the latest price
    ax.axhline(y=latest_price, color='r', linestyle='--', label=f'Latest Price: {latest_price:.2f}')
    
    # Annotate the latest price value on the chart
    ax.text(ax.get_xlim()[1], latest_price, f' {latest_price:.2f}', color='red', va='center')

    ax.legend()

    # Formatting the plot
    ax.set_title(f'Volume Mountain Map for {stock_code} (Time Granularity)')
    ax.set_xlabel('Cumulative Volume')
    ax.set_ylabel('Price')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # Ensure the plot directory exists
    plots_dir = 'd:\\stock\\plots_mountain_time'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Save the plot
    plt.savefig(os.path.join(plots_dir, f'mountain_profile_time_{stock_code}.png'))
    plt.close(fig)