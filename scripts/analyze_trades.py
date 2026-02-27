import json
import pandas as pd

def analyze_trade_reasons(log_file='backtest_detailed_log.json'):
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    trades = data.get('trade_records', [])
    trade_pairs = {}
    completed_trades = []
    
    for t in trades:
        code = t['code']
        if t['action'] == 'BUY_SIGNAL':
            trade_pairs[code] = {
                'buy_date': t['date'],
                'buy_price': t['price'],
                'buy_rank': t.get('ksp_rank', 'N/A')
            }
        elif t['action'] == 'SELL_SIGNAL':
            if code in trade_pairs:
                p = trade_pairs[code]
                pnl = (t['price'] - p['buy_price']) / p['buy_price']
                completed_trades.append({
                    'Code': code, 'In': p['buy_date'], 'Out': t['date'],
                    'PnL': f"{pnl:.2%}", 'Reason': t.get('reason', 'N/A')
                })
                del trade_pairs[code]

    for code, p in trade_pairs.items():
        completed_trades.append({'Code': code, 'In': p['buy_date'], 'Out': 'HOLDING', 'PnL': 'N/A', 'Reason': 'STILL HOLDING'})

    print("\n" + "="*80)
    print(f"{'Code':<10} | {'In':<10} | {'Out':<10} | {'PnL':<8} | {'Sell Reason'}")
    print("-" * 80)
    for r in completed_trades:
        print(f"{r['Code']:<10} | {r['In']:<10} | {r['Out']:<10} | {r['PnL']:<8} | {r['Reason']}")
    print("="*80)

if __name__ == "__main__":
    analyze_trade_reasons()
