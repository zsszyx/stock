import pandas as pd
import numpy as np
from datetime import datetime
from stock.data_context.context import Minutes5Context, DailyContext

def test_daily_context():
    # Create mock 5-minute data
    # 2 stocks, 2 days, 3 bars each day
    data = {
        'code': ['sh.600000']*6 + ['sz.000001']*6,
        'date': (['2024-03-01']*3 + ['2024-03-04']*3) * 2,
        'time': (['09:35:00', '09:40:00', '15:00:00']*2) * 2,
        'open': [10.0, 10.1, 10.2, 10.5, 10.4, 10.3] * 2,
        'high': [10.2, 10.3, 10.4, 10.6, 10.5, 10.4] * 2,
        'low': [9.9, 10.0, 10.1, 10.4, 10.3, 10.2] * 2,
        'close': [10.1, 10.2, 10.3, 10.4, 10.3, 10.2] * 2,
        'volume': [1000, 2000, 3000, 1500, 2500, 3500] * 2,
        'amount': [10100, 20400, 30900, 15600, 25750, 35700] * 2,
    }
    df = pd.DataFrame(data)
    
    min5_ctx = Minutes5Context(df)
    daily_ctx = DailyContext(min5_ctx)
    
    print("Daily Context Data:")
    print(daily_ctx.data)
    
    assert len(daily_ctx.data) == 4 # 2 codes * 2 days
    assert 'skew' in daily_ctx.data.columns
    assert 'kurt' in daily_ctx.data.columns
    assert 'poc' in daily_ctx.data.columns
    assert 'close' in daily_ctx.data.columns
    assert 'pct_chg' in daily_ctx.data.columns
    
    # Check close price for sh.600000 on 2024-03-01
    res = daily_ctx.data[(daily_ctx.data['code'] == 'sh.600000') & (daily_ctx.data['date'] == '2024-03-01')]
    assert res['close'].iloc[0] == 10.3
    
    # Check pct_chg for sh.600000 on 2024-03-04
    # (10.2 - 10.3) / 10.3 = -0.0097087...
    res_next = daily_ctx.data[(daily_ctx.data['code'] == 'sh.600000') & (daily_ctx.data['date'] == '2024-03-04')]
    np.testing.assert_almost_equal(res_next['pct_chg'].iloc[0], (10.2 - 10.3) / 10.3)
    
    print("Test passed!")

if __name__ == '__main__':
    test_daily_context()
