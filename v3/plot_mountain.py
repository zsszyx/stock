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
    
    # Create a single plot
    fig, ax = plt.subplots(figsize=(18, 10))

    # --- Main Scatter Plot ---
    # Prepare data for scatter plot
    plot_data = cumulative_pivot.stack().reset_index()
    plot_data.columns = ['price', 'time', 'cumulative_volume']
    plot_data = plot_data[plot_data['cumulative_volume'] > 0]

    # Use time for coloring. Newer times will have warmer colors with plasma.
    time_numeric = plot_data['time'].astype(np.int64)
    
    scatter = ax.scatter(plot_data['cumulative_volume'], plot_data['price'], c=time_numeric, cmap='plasma', s=10, alpha=0.7, zorder=2)

    # Get the latest price to use for clustering and reference
    latest_price = vp_df.sort_values(by='time', ascending=True).iloc[-1]['price']

    # --- Price Band Clustering and Background ---
    price_band_width = latest_price * 0.005
    final_cumulative_volume = cumulative_pivot.iloc[:, -1]
    
    # Create a dataframe from the final volumes
    final_volume_df = final_cumulative_volume.reset_index()
    final_volume_df.columns = ['price', 'volume']
    
    # Assign each price to a price band
    final_volume_df['price_band_center'] = (final_volume_df['price'] / price_band_width).round() * price_band_width
    
    # Group by the price band and sum the volumes
    clustered_data = final_volume_df.groupby('price_band_center')['volume'].sum().reset_index()

    # Use a color cycle for the bands
    cmap = plt.get_cmap('tab20')

    for i, row in clustered_data.iterrows():
        band_center = row['price_band_center']
        band_volume = row['volume']
        band_ymin = band_center - price_band_width / 2
        band_ymax = band_center + price_band_width / 2

        # Draw the background span
        ax.axhspan(band_ymin, band_ymax, color=cmap(i % cmap.N), alpha=0.2, zorder=0)

        # Display the total volume on the right
        ax.text(ax.get_xlim()[1] * 1.01, band_center, f'{band_volume:,.0f}',
                color='black', va='center', ha='left', fontsize=8)

    # --- Colorbar ---
    cbar = fig.colorbar(scatter, ax=ax)
    tick_values = cbar.get_ticks()
    cbar.set_ticks(tick_values)
    tick_labels = [pd.to_datetime(int(t)).strftime('%Y-%m-%d %H:%M') for t in tick_values]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Time')

    # --- Top 70% Volume Markers ---
    top_70_percent_threshold = final_cumulative_volume.quantile(0.3)
    top_prices = final_cumulative_volume[final_cumulative_volume >= top_70_percent_threshold]

    for price, volume in top_prices.items():
        ax.scatter(volume, price, color='blue', zorder=5)
        ax.text(volume, price, f' {volume:,.0f}', color='blue')

    # --- Latest Price Line ---
    ax.axhline(y=latest_price, color='r', linestyle='--', label=f'Latest Price: {latest_price:.2f}', zorder=3)
    ax.text(ax.get_xlim()[1], latest_price, f' {latest_price:.2f}', color='red', va='center')

    # --- Final Formatting ---
    ax.legend()
    ax.set_title(f'Volume Mountain Map for {stock_code} with Price Bands')
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