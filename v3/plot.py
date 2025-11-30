import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from datetime import datetime
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from utils import calculate_value_area_and_poc

def plot_volume_scatter(vp_df, stock_code, period_splits):
    """
    Plots the volume profile for a given stock as a scatter plot with time-based coloring
    and segmented Value Area backgrounds.

    Args:
        vp_df (pd.DataFrame): DataFrame with 'date', 'price', and 'total_volume'.
        stock_code (str): The stock code for labeling the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert 'date' to datetime objects
    vp_df['date'] = pd.to_datetime(vp_df['date'])
    
    # Sort by date to ensure correct color mapping
    vp_df = vp_df.sort_values(by='date')
    
    # Create a color map based on the date
    unique_dates = vp_df['date'].unique()
    colors = plt.cm.plasma(np.linspace(0, 1, len(unique_dates)))
    date_color_map = {date: color for date, color in zip(unique_dates, colors)}
    
    # Get the colors for each point
    point_colors = vp_df['date'].map(date_color_map)

    # Plotting the volume profile as a scatter plot
    ax.scatter(vp_df['total_volume'], vp_df['price'], c=point_colors, alpha=0.6, edgecolors='w', s=50)

    # Annotate the last 5 days
    if len(unique_dates) >= 5:
        last_5_dates = unique_dates[-5:]
        for i, date in enumerate(last_5_dates):
            date_df = vp_df[vp_df['date'] == date]
            for _, row in date_df.iterrows():
                ax.text(row['total_volume'], row['price'], str(i + 1), fontsize=8, color='black')

    # Aggregate volume by price for the line plot
    aggregated_vp = vp_df.groupby('price')['total_volume'].sum().reset_index()

    # Plot volume distribution and VA for 4 time segments
    if len(unique_dates) >= 4:
        segment_size = len(unique_dates) // 4
        segment_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'] # More distinct colors
        for i in range(4):
            start_index = i * segment_size
            end_index = (i + 1) * segment_size if i < 3 else len(unique_dates)
            segment_dates = unique_dates[start_index:end_index]
            segment_df = vp_df[vp_df['date'].isin(segment_dates)]
            
            if not segment_df.empty:
                # Calculate and display VA for the segment
                _, va_high, va_low = calculate_value_area_and_poc(segment_df)

    # Plot segmented volume distributions and their VAs
    segment_colors = ['#FF00FF', '#00FFFF', '#FFFF00', '#00FF00']  # Magenta, Cyan, Yellow, Green
    legend_elements_segments = []

    for i, period_dates in enumerate(period_splits):
        segment_df = vp_df[vp_df['date'].isin(period_dates)]
        if segment_df.empty:
            continue

        segment_vp = segment_df.groupby('price')['total_volume'].sum().reset_index()
        segment_vp.rename(columns={'total_volume': 'total_volume'}, inplace=True)

        poc, va_high, va_low = calculate_value_area_and_poc(segment_vp)

        color = segment_colors[i]

        if va_high is not None and va_low is not None:
            ax.axhline(poc, color=color, linestyle='-', linewidth=1.2, alpha=0.9)

        start_period_date = period_dates[0].strftime('%Y-%m-%d')
        end_period_date = period_dates[-1].strftime('%Y-%m-%d')
        legend_elements_segments.append(Line2D([0], [0], color=color, label=f'Seg {i+1} POC: {start_period_date} to {end_period_date}'))

    # Final plot adjustments and legend creation
    ax.set_xlabel('Total Volume')
    ax.set_ylabel('Price')
    ax.set_title(f'Time-based Volume Profile for {stock_code}')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Calculate and plot the overall POC
    overall_poc, _, _ = calculate_value_area_and_poc(aggregated_vp)
    if overall_poc is not None:
        ax.axhline(overall_poc, color='blue', linestyle=':', linewidth=1.5, label=f'Overall POC: {overall_poc:.2f}')

    # Create a comprehensive custom legend
    handles, labels = ax.get_legend_handles_labels()
    
    legend_elements = []
    if len(unique_dates) > 1:
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', label=f'Earliest: {pd.to_datetime(unique_dates[0]).strftime("%Y-%m-%d")}', markerfacecolor=colors[0], markersize=10),
            Line2D([0], [0], marker='o', color='w', label=f'Latest: {pd.to_datetime(unique_dates[-1]).strftime("%Y-%m-%d")}', markerfacecolor=colors[-1], markersize=10)
        ])
    
    legend_elements.extend(legend_elements_segments)
    
    legend_elements.append(Line2D([0], [0], color='blue', linestyle=':', lw=1.5, label='Overall POC'))

    ax.legend(handles=handles + legend_elements, title="Legend", loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Ensure the plot directory exists
    plots_dir = 'd:\\stock\\plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Save the plot
    plt.savefig(os.path.join(plots_dir, f'volume_profile_{stock_code}.png'))
    plt.close()

if __name__ == '__main__':
    try:
        all_vps = pd.read_csv('d:\\stock\\volume_profiles.csv')
    except FileNotFoundError:
        print("Error: 'volume_profiles.csv' not found. Please run vp.py first.")
        exit()

    # Group by stock code and plot each one
    for code, group in all_vps.groupby('code'):
        plot_volume_scatter(group, code)

    print("Plots saved in the 'plots' directory.")