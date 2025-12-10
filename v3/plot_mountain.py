import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_volume_mountain(vp_df, stock_code):
    """
    Plots a volume mountain chart and a volume growth chart for a given stock.

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
    
    # Create a figure with two subplots (side-by-side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), gridspec_kw={'width_ratios': [3, 1]})

    # --- Subplot 1: Main Scatter Plot (Volume Mountain) ---
    plot_data = cumulative_pivot.stack().reset_index()
    plot_data.columns = ['price', 'time', 'cumulative_volume']
    plot_data = plot_data[plot_data['cumulative_volume'] > 0]

    time_numeric = plot_data['time'].astype(np.int64)
    
    scatter = ax1.scatter(plot_data['cumulative_volume'], plot_data['price'], c=time_numeric, cmap='plasma', s=10, alpha=0.7, zorder=2)

    latest_price = vp_df.sort_values(by='time', ascending=True).iloc[-1]['price']
    price_band_width = latest_price * 0.005
    final_cumulative_volume = cumulative_pivot.iloc[:, -1]
    
    final_volume_df = final_cumulative_volume.reset_index()
    final_volume_df.columns = ['price', 'volume']
    final_volume_df['price_band_center'] = (final_volume_df['price'] / price_band_width).round() * price_band_width
    
    clustered_data = final_volume_df.groupby('price_band_center')['volume'].sum().reset_index()

    cmap = plt.get_cmap('tab20')

    for i, row in clustered_data.iterrows():
        band_center = row['price_band_center']
        band_volume = row['volume']
        band_ymin = band_center - price_band_width / 2
        band_ymax = band_center + price_band_width / 2

        ax1.axhspan(band_ymin, band_ymax, color=cmap(i % cmap.N), alpha=0.2, zorder=0)
        ax1.text(ax1.get_xlim()[1] * 1.01, band_center, f'{band_volume:,.0f}', color='black', va='center', ha='left', fontsize=8)

    cbar = fig.colorbar(scatter, ax=ax1)
    tick_values = cbar.get_ticks()
    cbar.set_ticks(tick_values)
    tick_labels = [pd.to_datetime(int(t)).strftime('%Y-%m-%d %H:%M') for t in tick_values]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Time')

    top_70_percent_threshold = final_cumulative_volume.quantile(0.3)
    top_prices = final_cumulative_volume[final_cumulative_volume >= top_70_percent_threshold]

    for price, volume in top_prices.items():
        ax1.scatter(volume, price, color='blue', zorder=5)
        ax1.text(volume, price, f' {volume:,.0f}', color='blue')

    ax1.axhline(y=latest_price, color='r', linestyle='--', label=f'Latest Price: {latest_price:.2f}', zorder=3)
    ax1.text(ax1.get_xlim()[1], latest_price, f' {latest_price:.2f}', color='red', va='center')

    ax1.legend()
    ax1.set_title(f'Volume Mountain Map for {stock_code}')
    ax1.set_xlabel('Cumulative Volume')
    ax1.set_ylabel('Price')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Subplot 2: Volume Increment in Current Price Band ---
    current_band_center = (latest_price / price_band_width).round() * price_band_width
    current_band_prices = final_volume_df[final_volume_df['price_band_center'] == current_band_center]['price']
    
    band_pivot = pivot_df[pivot_df.index.isin(current_band_prices)]
    band_time_series = band_pivot.sum(axis=0)
    
    if not band_time_series.empty:
        time_points = band_time_series.index
        volume_points = band_time_series.values

        ax2.scatter(time_points, volume_points, label='Volume Increment in Current Band')

        # Fit a curve (e.g., polynomial)
        time_numeric_for_fit = (time_points - time_points.min()).total_seconds()
        coeffs = np.polyfit(time_numeric_for_fit, volume_points, 2) # 2nd degree polynomial
        poly = np.poly1d(coeffs)
        
        smooth_time = np.linspace(time_numeric_for_fit.min(), time_numeric_for_fit.max(), 300)
        smooth_volume = poly(smooth_time)
        smooth_time_dt = time_points.min() + pd.to_timedelta(smooth_time, unit='s')

        ax2.plot(smooth_time_dt, smooth_volume, color='red', linestyle='--', label='Fitted Growth Curve')

    ax2.set_title(f'Volume Increment at {current_band_center:.2f} Price Band')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volume Increment in Band')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    # --- Final Layout Adjustments ---
    plt.tight_layout(rect=[0, 0, 0.95, 1])

    plots_dir = 'd:\\stock\\plots_mountain_time'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.savefig(os.path.join(plots_dir, f'mountain_profile_time_{stock_code}.png'))
    plt.close(fig)