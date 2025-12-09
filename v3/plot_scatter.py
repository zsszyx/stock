import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_volume_scatter(df, stock_code):
    """
    Plots a volume scatter plot where the x-axis is time, the y-axis is price,
    and the color of the points represents the volume.

    Args:
        df (pd.DataFrame): DataFrame with 'time', 'price', and 'total_volume'.
        stock_code (str): The stock code for labeling the plot.
    """
    if df.empty:
        print(f"No data for stock {stock_code}, skipping plot.")
        return

    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M%S%f')
    df = df.sort_values(by=['price', 'time'])
    
    # Identify consecutive points with the same price
    df['price_group'] = (df['price'] != df['price'].shift()).cumsum()
    
    # Aggregate the data
    agg_df = df.groupby('price_group').agg(
        time=('time', lambda x: x.iloc[0] + (x.iloc[-1] - x.iloc[0]) / 2),
        price=('price', 'first'),
        total_volume=('total_volume', 'sum')
    ).reset_index()

    # Use qcut to create bins based on the distribution of 'total_volume'
    try:
        agg_df['volume_cat'] = pd.qcut(agg_df['total_volume'], q=10, labels=False, duplicates='drop')
    except ValueError:
        # If qcut fails (e.g., not enough unique values), use rank
        agg_df['volume_cat'] = agg_df['total_volume'].rank(method='dense').astype(int)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use the new 'volume_cat' for coloring
    scatter = ax.scatter(agg_df['time'], agg_df['price'], c=agg_df['volume_cat'], cmap='hot', s=10)
    
    # Add a colorbar to show the volume scale
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Volume')
    
    # Formatting the plot
    ax.set_title(f'Volume Scatter Plot for {stock_code}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Ensure the plot directory exists
    plots_dir = 'd:\\stock\\plots_scatter'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    # Save the plot
    plt.savefig(os.path.join(plots_dir, f'scatter_profile_{stock_code}.png'))
    plt.close(fig)