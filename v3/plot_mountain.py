import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_volume_mountain(vp_df, stock_code, volatility):
    """
    Plots a volume mountain chart for a given stock, with price bands based on volatility.

    Args:
        vp_df (pd.DataFrame): DataFrame with 'time', 'price', and 'total_volume'.
        stock_code (str): The stock code for labeling the plot.
        volatility (float): The price volatility to determine band width.
    """
    if vp_df.empty:
        print(f"No data for stock {stock_code}, skipping plot.")
        return

    vp_df['time'] = pd.to_datetime(vp_df['time'], format='%Y%m%d%H%M%S%f')
    
    # Pivot to get prices on rows, time on columns
    pivot_df = vp_df.pivot_table(index='price', columns='time', values='total_volume', fill_value=0)
    
    # Ensure columns (timestamps) are sorted before cumulative sum
    pivot_df = pivot_df.sort_index(axis=1)
    
    # Calculate cumulative sum across time (columns)
    cumulative_pivot = pivot_df.cumsum(axis=1)
    
    # Create a figure with two subplots (side-by-side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), gridspec_kw={'width_ratios': [3, 1]})

    # --- Subplot 1: Main Scatter Plot (Volume Mountain) ---
    plot_data = cumulative_pivot.stack().reset_index()
    plot_data.columns = ['price', 'time', 'cumulative_volume']
    plot_data = plot_data[plot_data['cumulative_volume'] > 0]

    # Sort by time to ensure correct layering with alpha blending
    plot_data = plot_data.sort_values('time')

    time_numeric = plot_data['time'].astype(np.int64)
    
    scatter = ax1.scatter(plot_data['cumulative_volume'], plot_data['price'], c=time_numeric, cmap='plasma', s=10, alpha=0.7, zorder=2)

    latest_price = vp_df.sort_values(by='time', ascending=True).iloc[-1]['price']
    price_band_width = volatility  # Use volatility for band width
    final_cumulative_volume = cumulative_pivot.iloc[:, -1]
    
    final_volume_df = final_cumulative_volume.reset_index()
    final_volume_df.columns = ['price', 'volume']
    final_volume_df['price_band_center'] = (final_volume_df['price'] / price_band_width).round() * price_band_width
    
    clustered_data = final_volume_df.groupby('price_band_center')['volume'].sum().reset_index()

    cmap = plt.get_cmap('tab20')

    # Draw background bands on the main plot
    for i, row in clustered_data.iterrows():
        band_center = row['price_band_center']
        band_ymin = band_center - price_band_width / 2
        band_ymax = band_center + price_band_width / 2
        ax1.axhspan(band_ymin, band_ymax, color=cmap(i % cmap.N), alpha=0.2, zorder=0)

    cbar = fig.colorbar(scatter, ax=ax1)
    tick_values = cbar.get_ticks()
    cbar.set_ticks(tick_values)
    tick_labels = [pd.to_datetime(int(t)).strftime('%Y-%m-%d %H:%M') for t in tick_values]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Time')

    ax1.axhline(y=latest_price, color='r', linestyle='--', label=f'Latest Price: {latest_price:.2f}', zorder=3)
    ax1.text(ax1.get_xlim()[1], latest_price, f' {latest_price:.2f}', color='red', va='center')

    ax1.legend()
    ax1.set_title(f'Volume Mountain Map for {stock_code}')
    ax1.set_xlabel('Cumulative Volume')
    ax1.set_ylabel('Price')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Subplot 2: Volume by Price Band ---
    band_centers = clustered_data['price_band_center']
    band_volumes = clustered_data['volume']
    colors = [cmap(i % cmap.N) for i in range(len(clustered_data))]

    ax2.barh(band_centers, band_volumes, color=colors, height=price_band_width, alpha=0.6)
    
    # Align y-axes
    ax2.set_ylim(ax1.get_ylim())
    ax2.get_yaxis().set_visible(False) # Hide y-axis labels to avoid clutter

    ax2.set_title('Volume by Price Band')
    ax2.set_xlabel('Total Volume')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add volume labels to the bars
    for index, value in enumerate(band_volumes):
        ax2.text(value, band_centers.iloc[index], f' {value:,.0f}', va='center')

    # --- Final Layout Adjustments ---
    plt.tight_layout(rect=[0, 0, 0.95, 1])

    plots_dir = 'd:\\stock\\plots_mountain_time'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.savefig(os.path.join(plots_dir, f'mountain_profile_time_{stock_code}.png'))
    plt.close(fig)