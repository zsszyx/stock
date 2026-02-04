import pandas as pd
import numpy as np
from datetime import datetime
from stock.data_context.context import Minutes5Context, DailyContext
from stock.selector.simple_selector import RecentDaysPctChgSelector, POCNearSelector

def test_selector():
    # Create mock 5-minute data for 6 days to have 5 days of pct_chg
    dates = ['2024-03-01', '2024-03-04', '2024-03-05', '2024-03-06', '2024-03-07', '2024-03-08']
    codes = ['sh.600000', 'sz.000001']
    
    data_list = []
    for code in codes:
        for i, date in enumerate(dates):
            # sh.600000: small changes, within 5%
            # sz.000001: large changes, exceeding 5%
            if code == 'sh.600000':
                price = 10.0 + (i * 0.01) # 10.00 -> 10.05 (0.5% total)
            else:
                price = 10.0 + (i * 0.2) # 10.0 -> 11.0 (10% total)
                
            data_list.append({
                'code': code,
                'date': date,
                'time': '15:00:00',
                'open': price - 0.1,
                'high': price + 0.1,
                'low': price - 0.1,
                'close': price,
                'volume': 1000,
                'amount': price * 1000
            })
            
    df = pd.DataFrame(data_list)
    min5_ctx = Minutes5Context(df)
    daily_ctx = DailyContext(min5_ctx)
    
    # --- Test RecentDaysPctChgSelector ---
    selector = RecentDaysPctChgSelector(daily_ctx)
    target_date = datetime.strptime('2024-03-08', '%Y-%m-%d')
    selected = selector.select(target_date, days=5, threshold=0.05)
    
    print(f"Daily Data:\n{daily_ctx.data[['code', 'date', 'close', 'poc', 'pct_chg']]}")
    print(f"PctChg Selected codes: {selected}")
    
    assert 'sh.600000' in selected
    assert 'sz.000001' not in selected
    
    # --- Test POCNearSelector ---
    poc_selector = POCNearSelector(daily_ctx)
    # Our mock data for each day only has one 5-min bar for simplicity in some days, 
    # and POC will be exactly the price of that bar.
    selected_poc = poc_selector.select(target_date, threshold=0.01)
    print(f"POC Selected codes: {selected_poc}")
    
    assert 'sh.600000' in selected_poc
    assert 'sz.000001' in selected_poc
    
    print("Selector test passed!")

if __name__ == '__main__':
    test_selector()