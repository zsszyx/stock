import json
import pandas as pd
from datetime import datetime

def audit_backtest_logs(log_file='backtest_detailed_log.json'):
    print(f"审计 {log_file}...")
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    trades = data.get('trade_records', [])
    daily = data.get('daily_records', [])
    anomalies = []

    for record in daily:
        if record['position_count'] > 9:
            anomalies.append(f"Position overflow: {record['date']} count={record['position_count']}")
        if record['total_value'] <= 0:
            anomalies.append(f"Negative value: {record['date']} val={record['total_value']}")

    holding_status = {}
    for t in trades:
        code, date, action = t['code'], t['date'], t['action']
        if action == 'BUY_SIGNAL':
            if code in holding_status:
                anomalies.append(f"Double Buy: {date} {code}")
            holding_status[code] = date
        elif action == 'SELL_SIGNAL':
            if code not in holding_status:
                anomalies.append(f"Orphan Sell: {date} {code}")
            else:
                del holding_status[code]

    print("\n" + "="*30 + "\nAudit Report\n" + "="*30)
    if not anomalies:
        print("✅ Success: No anomalies found.")
    else:
        for a in anomalies: print(f"🚨 {a}")
    
    if daily:
        last = daily[-1]
        print(f"\nEnd State: {last['date']}")
        print(f"Total Value: {last['total_value']:,.2f}")
        for p in last['positions']:
            print(f"- {p['code']}: {p['profit_pct']:.2%}")

if __name__ == "__main__":
    audit_backtest_logs()
