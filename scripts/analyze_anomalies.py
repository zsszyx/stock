import json
import pandas as pd
from datetime import datetime

def analyze_backtest_log(json_path='backtest_detailed_log.json'):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    daily_records = data.get('daily_records', [])
    trade_records = data.get('trade_records', [])
    
    df_daily = pd.DataFrame(daily_records)
    df_trades = pd.DataFrame(trade_records)
    
    # 转换为时间序列
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.sort_values('date')
    
    # 计算价值变化
    df_daily['value_change'] = df_daily['total_value'].diff().abs()
    
    # 查找长时间价值不变的时期 (连续 5 天以上变化接近 0)
    df_daily['is_flat'] = df_daily['value_change'] < 1e-4
    df_daily['flat_group'] = (df_daily['is_flat'] != df_daily['is_flat'].shift()).cumsum()
    
    flat_periods = []
    for group_id, group in df_daily[df_daily['is_flat']].groupby('flat_group'):
        if len(group) >= 5:
            start_date = group['date'].min()
            end_date = group['date'].max()
            flat_periods.append({
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'days': len(group),
                'value': group['total_value'].iloc[0],
                'cash': group['cash'].iloc[0],
                'pos_count': group['position_count'].iloc[0]
            })
            
    print(f"🔍 发现 {len(flat_periods)} 段收益平坦期 (>= 5天):")
    for p in flat_periods:
        print(f"📅 {p['start']} 至 {p['end']} | 天数: {p['days']} | 持仓数: {p['pos_count']} | 现金: {p['cash']:,.2f}")
        
        # 分析该段时期的交易记录
        period_trades = df_trades[(pd.to_datetime(df_trades['date']) >= pd.to_datetime(p['start'])) & 
                                  (pd.to_datetime(df_trades['date']) <= pd.to_datetime(p['end']))]
        if period_trades.empty:
            print("   ⚠️ 该期间无任何买卖操作")
        else:
            print(f"   📑 该期间操作记录: {len(period_trades)} 笔")
            print(period_trades[['date', 'action', 'code', 'price']].to_string(index=False))
            
    # 输出详细日报到 CSV
    df_daily.to_csv('output/daily_operation_analysis.csv', index=False)
    print(f"\n✅ 详细日报已保存至: output/daily_operation_analysis.csv")

if __name__ == "__main__":
    analyze_backtest_log()
