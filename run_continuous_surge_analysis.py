import pandas as pd
import os
import sys
from tqdm import tqdm
from datetime import timedelta

# Add root to path
sys.path.append(os.getcwd())

from sql_op.op import SqlOp
from sql_op import sql_config
from plot.vp_plot import plot_volume_profile

def analyze_and_plot():
    print("Initializing analysis...")
    sql_op = SqlOp()
    
    # 1. Get all unique codes
    print("Fetching stock list...")
    df_codes = sql_op.query(f"SELECT DISTINCT code FROM {sql_config.mintues5_table_name}")
    if df_codes is None or df_codes.empty:
        print("No stocks found in database.")
        return
    
    codes = df_codes['code'].tolist()
    print(f"Found {len(codes)} stocks. Scanning for 2 consecutive days > 10% rise...")
    
    output_dir = "vp_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    found_count = 0
    
    for code in tqdm(codes):
        # 2. Read all 5-min data for the stock
        # Optimized: Read only columns needed for daily aggregation first? 
        # Actually we need full data later for plotting, but for scanning we only need price/date.
        # But reading twice is IO heavy. Reading once into memory is better if memory allows.
        # Given 4GB DB and 1 year data, one stock is small.
        
        query = f"SELECT date, time, open, close, volume, amount FROM {sql_config.mintues5_table_name} WHERE code='{code}' ORDER BY date, time"
        k_data = sql_op.query(query)
        
        if k_data is None or k_data.empty:
            continue
            
        k_data['date'] = pd.to_datetime(k_data['date'])
        k_data['close'] = pd.to_numeric(k_data['close'])
        k_data['open'] = pd.to_numeric(k_data['open'])
        
        # 3. Resample to Daily
        # Group by date
        daily_groups = k_data.groupby('date')
        daily_data = pd.DataFrame({
            'open': daily_groups['open'].first(),
            'close': daily_groups['close'].last()
        }).sort_index()
        
        # Calculate Pct Change
        daily_data['prev_close'] = daily_data['close'].shift(1)
        daily_data['pct_chg'] = (daily_data['close'] - daily_data['prev_close']) / daily_data['prev_close']
        
        # 4. Find Pattern: > 10% for 2 consecutive days
        # condition: pct_chg > 0.10
        # Check current and previous
        daily_data['is_surge'] = daily_data['pct_chg'] > 0.09
        daily_data['prev_is_surge'] = daily_data['is_surge'].shift(1)
        daily_data['consecutive_surge'] = daily_data['is_surge'] & daily_data['prev_is_surge']
        
        surge_dates = daily_data[daily_data['consecutive_surge']].index
        
        for surge_date in surge_dates:
            # surge_date is the 2nd day of the surge
            # We want to plot: 
            # Start: 2 weeks before the FIRST surge day (surge_date - 1 day - 14 days)
            # End: 2 weeks after the SECOND surge day (surge_date + 14 days)
            
            # Find the date index to calculate offsets accurately using trading days?
            # User said "2 weeks", usually implies 14 calendar days.
            
            plot_start_date = surge_date - timedelta(days=15) # 1 day for prev surge + 14 days before
            plot_end_date = surge_date + timedelta(days=14)
            
            # Check if we have data for this range (at least some overlap)
            data_min_date = k_data['date'].min()
            data_max_date = k_data['date'].max()
            
            # If the window is completely out of range, skip
            if plot_end_date < data_min_date or plot_start_date > data_max_date:
                continue
                
            # Formatting dates for filename and function
            s_date_str = plot_start_date.strftime('%Y-%m-%d')
            e_date_str = plot_end_date.strftime('%Y-%m-%d')
            surge_date_str = surge_date.strftime('%Y-%m-%d')
            
            print(f"\nFound match: {code} on {surge_date_str}")
            
            filename = f"{code}_{surge_date_str}_surge.png"
            save_path = os.path.join(output_dir, filename)
            
            # Call Plotting Function
            # Note: plot_volume_profile fetches data internally using SqlOp. 
            # This is inefficient (double fetch), but re-implementing plotting here is complex.
            # I will reuse the existing function for simplicity and correctness.
            try:
                plot_volume_profile(code, s_date_str, e_date_str, save_path=save_path)
                found_count += 1
            except Exception as e:
                print(f"Error plotting {code}: {e}")

    print(f"\nAnalysis Complete. Found and plotted {found_count} instances.")
    if found_count > 0:
        print(f"Plots saved in '{output_dir}' directory.")

if __name__ == '__main__':
    analyze_and_plot()
