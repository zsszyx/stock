import json
import pandas as pd
from collections import defaultdict

def calculate_avg_holding_days(log_path):
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    trade_records = data.get('trade_records', [])
    active_buys = defaultdict(list)
    holding_days_list = []
    
    for rec in trade_records:
        code = rec['code']
        if rec['action'] == 'BUY_FILL':
            active_buys[code].append(rec['date'])
        elif rec['action'] == 'SELL_FILL':
            if active_buys[code]:
                buy_date = active_buys[code].pop(0)
                days = (pd.to_datetime(rec['date']) - pd.to_datetime(buy_date)).days
                holding_days_list.append(days)
    
    if not holding_days_list:
        return 0
    
    return sum(holding_days_list) / len(holding_days_list)

if __name__ == "__main__":
    log_file = "logs/KSP_V7_SUBOPTIMAL_FULL_20260226_213602.json"
    avg_days = calculate_avg_holding_days(log_file)
    print(f"Average Holding Days: {avg_days:.2f}")
