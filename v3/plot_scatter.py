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
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(df['time'], df['price'], c=df['total_volume'], cmap='hot', s=10)
    
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