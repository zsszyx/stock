import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sql_op.op import SqlOp
from sql_op import sql_config

def plot_volume_profile(code, start_date, end_date, price_bins=50, save_path=None):
    """
    Plots a Volume Profile (VP) chart where the X-axis is time (days), 
    and for each day, a volume profile histogram is drawn vertically.
    
    Args:
        code (str): Stock code.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        price_bins (int): Number of price bins for the histogram.
        save_path (str, optional): Path to save the image. If None, shows the plot.
    """
    sql_operator = SqlOp()
    
    # Fetch 5-minute data
    print(f"Fetching data for {code} from {start_date} to {end_date}...")
    k_data = sql_operator.read_k_data_by_date_range(sql_config.mintues5_table_name, start_date, end_date)
    
    if k_data.empty:
        print("No data found.")
        return

    # Filter for the specific code (if the table contains multiple)
    # The read_k_data_by_date_range gets all codes, so we filter in memory or should upgrade the SQL op.
    # For now, filter in memory as per existing pattern.
    # Normalize code check: input '600000', db might be 'sh.600000' or '600000'.
    # Let's assume input is simple code, DB has prefixes or simple codes.
    # Check first row to see DB format.
    sample_code = k_data['code'].iloc[0]
    if '.' in str(sample_code):
        # DB has prefix, check if input has prefix
        if '.' not in code:
            # simple input, match suffix
            k_data = k_data[k_data['code'].str.endswith(code)]
        else:
            k_data = k_data[k_data['code'] == code]
    else:
        # DB has no prefix
        if '.' in code:
            code = code.split('.')[-1]
        k_data = k_data[k_data['code'] == code]

    if k_data.empty:
        print(f"No data found for code {code}.")
        return

    # Process Data
    k_data['date'] = pd.to_datetime(k_data['date'])
    cols_to_numeric = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in cols_to_numeric:
        k_data[col] = pd.to_numeric(k_data[col], errors='coerce')
    
    k_data = k_data[k_data['volume'] > 0]
    # Use average price of the 5-min bar as the 'price' for volume accumulation
    k_data['avg_price'] = k_data['amount'] / k_data['volume']

    # Setup Plot: 2 Subplots (Top for VP, Bottom for Skew/Kurt)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Get unique sorted dates
    dates = sorted(k_data['date'].unique())
    
    # Map dates to X-axis indices (0, 1, 2...) to ensure equal spacing
    # regardless of gaps (weekends/holidays)
    date_map = {date: i for i, date in enumerate(dates)}
    
    # Track min/max price for Y-axis scaling
    y_min, y_max = k_data['avg_price'].min(), k_data['avg_price'].max()
    
    # Fetch daily stats (Skewness/Kurtosis)
    print("Fetching daily stats...")
    stats_query = f"""
        SELECT date, weighted_skew, weighted_kurt 
        FROM {sql_config.daily_stats_table_name} 
        WHERE code = '{code}' AND date >= '{start_date}' AND date <= '{end_date}'
    """
    stats_df = sql_operator.query(stats_query)
    daily_stats_map = {}
    if stats_df is not None and not stats_df.empty:
        # Normalize date to string or timestamp for matching
        stats_df['date'] = pd.to_datetime(stats_df['date'])
        for _, row in stats_df.iterrows():
            daily_stats_map[row['date']] = {
                'skew': row['weighted_skew'],
                'kurt': row['weighted_kurt']
            }
    
    # Width of each day's slot (e.g., 0.8 means 80% of the space between days)
    slot_width = 0.8 
    
    print(f"Plotting {len(dates)} days...")

    for date in dates:
        day_data = k_data[k_data['date'] == date]
        if day_data.empty:
            continue
            
        x_center = date_map[date]
        
        # Create Histogram
        # We weigh the histogram by volume
        hist, bin_edges = np.histogram(
            day_data['avg_price'], 
            bins=price_bins, 
            weights=day_data['volume'],
            range=(day_data['avg_price'].min(), day_data['avg_price'].max())
        )
        
        # Normalize histogram to fit in the slot_width
        # Max volume in this day = slot_width
        if hist.max() > 0:
            hist_norm = (hist / hist.max()) * slot_width
        else:
            hist_norm = hist

        # Plot bars (horizontal) on Top Subplot (ax1)
        # bin_edges has size bins+1. We need centers.
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        height = bin_edges[1] - bin_edges[0]
        
        left = x_center - slot_width / 2
        ax1.barh(bin_centers, hist_norm, height=height, left=left, color='skyblue', alpha=0.7, edgecolor='grey', linewidth=0.5)
        
        # Calculate POC (Point of Control) - Price with max volume
        if len(hist) > 0:
            max_vol_idx = np.argmax(hist)
            poc_price = bin_centers[max_vol_idx]
            # Plot POC
            ax1.scatter(x_center, poc_price, color='red', s=15, zorder=10, label='POC' if date == dates[0] else "")
            
        # Annotation for Skew/Kurt on Top Subplot (Removed as per user request)
        # if date in daily_stats_map:
        #     stats = daily_stats_map[date]
        #     text_str = f"S:{stats['skew']:.2f}\nK:{stats['kurt']:.2f}"
        #     ax1.text(x_center, y_max, text_str, ha='center', va='bottom', fontsize=8, color='darkblue')

    # Formatting Top Subplot (ax1)
    ax1.set_title(f"Volume Profile by Date for {code} ({start_date} - {end_date})")
    ax1.set_ylabel("Price")
    ax1.set_xlim(-1, len(dates))
    ax1.set_ylim(y_min * 0.95, y_max * 1.05)
    
    # --- Plot Skewness/Kurtosis on Bottom Subplot (ax2) ---
    skew_values = []
    kurt_values = []
    valid_indices = []
    
    for i, date in enumerate(dates):
        if date in daily_stats_map:
            skew_values.append(daily_stats_map[date]['skew'])
            kurt_values.append(daily_stats_map[date]['kurt'])
            valid_indices.append(i)
    
    if valid_indices:
        # Plot Skewness
        ax2.plot(valid_indices, skew_values, color='orange', linestyle='-', marker='o', markersize=4, label='S (Skewness)')
        # Plot Kurtosis
        ax2.plot(valid_indices, kurt_values, color='green', linestyle='--', marker='x', markersize=4, label='K (Kurtosis)')
        
        # Add text annotations for each data point
        for i, idx in enumerate(valid_indices):
            # Skewness
            ax2.text(idx, skew_values[i], f"{skew_values[i]:.2f}", 
                     ha='center', va='bottom', fontsize=7, color='orange')
            # Kurtosis
            ax2.text(idx, kurt_values[i], f"{kurt_values[i]:.2f}", 
                     ha='center', va='top', fontsize=7, color='green')

        # Add Legend to Bottom Subplot
        ax2.legend(loc='upper right')

    # Add Legend to Top Subplot (for POC)
    ax1.legend(loc='upper right')

    # Set X-ticks (shared)
    # Show every Nth date to avoid clutter
    num_ticks = 10
    step = max(1, len(dates) // num_ticks)
    # Set ticks on the shared axis (bottom one usually controls the view)
    ax2.set_xticks([i for i in range(0, len(dates), step)])
    ax2.set_xticklabels([dates[i].strftime('%Y-%m-%d') for i in range(0, len(dates), step)], rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == '__main__':
    # Test
    plot_volume_profile('002939', '2026-01-01', '2026-01-22')
