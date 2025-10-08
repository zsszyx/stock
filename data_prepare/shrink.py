import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 参数配置 (Configuration) ---
# 这些参数可以根据不同市场、不同板块的特性进行调整
CONFIG = {
    # 成交量维度参数
    'volume_long_period': 120,     # 长期成交量均线周期 (e.g., 半年线)
    'volume_short_period': 10,     # 短期成交量均线周期
    'volume_ratio_threshold': 0.5, # 短期均量低于长期均量的阈值 (e.g., 50%)
    'volume_quiet_days': 10,       # 成交量低迷需要持续的天数

    # 波动率维度参数
    'bb_period': 20,               # 布林带计算周期
    'bb_std_dev': 2,               # 布林带标准差倍数
    'volatility_lookback': 252,    # 波动率历史回看周期 (e.g., 一年)
    'volatility_percentile': 0.1,  # 波动率指标处于历史区间的百分位阈值 (e.g., 最低的10%)

    # 动能维度参数
    'momentum_period': 20,         # 价格动能考察周期
    'momentum_range_percentile': 0.1 # 价格振幅处于历史区间的百分位阈值
}

def analyze_stock_fragility(data, ticker: str, config: dict) -> dict:
    """
    分析单只股票是否处于“脆弱共识”阶段的核心函数
    
    :param ticker: 股票代码
    :param config: 参数配置字典
    :return: 包含分析结果和数据的字典
    """

    # --- 2. 计算各项量化指标 ---

    # A. 成交量维度指标
    data['volume_long_ma'] = data['Volume'].rolling(window=config['volume_long_period']).mean()
    data['volume_short_ma'] = data['Volume'].rolling(window=config['volume_short_period']).mean()
    # 条件1: 短期均量是否低于长期均量的一定比例
    data['volume_condition_met'] = data['volume_short_ma'] < (data['volume_long_ma'] * config['volume_ratio_threshold'])

    # B. 波动率维度指标 (使用布林带宽度 Bollinger Band Width, BBW)
    mid_band = data['Close'].rolling(window=config['bb_period']).mean()
    std_dev = data['Close'].rolling(window=config['bb_period']).std()
    upper_band = mid_band + (config['bb_std_dev'] * std_dev)
    lower_band = mid_band - (config['bb_std_dev'] * std_dev)
    data['bbw'] = (upper_band - lower_band) / mid_band
    # 条件2: 当前波动率是否处于历史极低水平
    # 计算BBW在过去volatility_lookback天内的百分位排名
    data['bbw_percentile'] = data['bbw'].rolling(window=config['volatility_lookback']).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # C. 动能维度指标 (使用价格振幅)
    price_range = data['High'].rolling(window=config['momentum_period']).max() - data['Low'].rolling(window=config['momentum_period']).min()
    avg_price = data['Close'].rolling(window=config['momentum_period']).mean()
    data['normalized_range'] = price_range / avg_price
    # 条件3: 当前价格振幅是否处于历史极低水平
    data['range_percentile'] = data['normalized_range'].rolling(window=config['volatility_lookback']).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    # --- 3. 综合判断 ---
    
    # 获取最近一个交易日的数据
    latest = data.iloc[-1]
    
    # 判断成交量条件是否【持续】满足
    is_volume_quiet_sustained = data['volume_condition_met'].tail(config['volume_quiet_days']).all()
    
    # 判断波动率和动能条件是否满足
    is_volatility_compressed = latest['bbw_percentile'] < config['volatility_percentile']
    is_momentum_stalled = latest['range_percentile'] < config['momentum_range_percentile']

    # 最终诊断
    is_fragile = is_volume_quiet_sustained and is_volatility_compressed and is_momentum_stalled
    
    # 准备返回结果
    result = {
        'ticker': ticker,
        'is_fragile': is_fragile,
        'details': {
            'volume_sustained_quiet': {
                'status': bool(is_volume_quiet_sustained),
                'short_ma': f"{latest['volume_short_ma']:.0f}",
                'long_ma': f"{latest['volume_long_ma']:.0f}",
                'ratio': f"{(latest['volume_short_ma'] / latest['volume_long_ma']):.2%}" if latest['volume_long_ma'] > 0 else "N/A"
            },
            'volatility_compressed': {
                'status': bool(is_volatility_compressed),
                'current_bbw_percentile': f"{latest['bbw_percentile']:.2%}" if pd.notna(latest['bbw_percentile']) else "N/A"
            },
            'momentum_stalled': {
                'status': bool(is_momentum_stalled),
                'current_range_percentile': f"{latest['range_percentile']:.2%}" if pd.notna(latest['range_percentile']) else "N/A"
            }
        }
    }
    
    return result

# --- 执行扫描 ---
if __name__ == '__main__':
    # 设定你想要扫描的股票列表
    # 示例: 包含不同类型的知名美股
    stock_list = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'PFE', 'INTC', 'BABA']
    
    print(f"--- 开始扫描 {len(stock_list)} 只股票 ---")
    
    fragile_stocks = []
    
    for stock in stock_list:
        analysis_result = analyze_stock_fragility(stock, CONFIG)
        if analysis_result['is_fragile']:
            fragile_stocks.append(analysis_result)
            print(f"✅ [发现!] {stock} 可能处于脆弱共识阶段。")
            print(f"   - 成交量持续低迷: {analysis_result['details']['volume_sustained_quiet']['status']}")
            print(f"   - 波动率极度压缩: {analysis_result['details']['volatility_compressed']['status']} (历史百分位: {analysis_result['details']['volatility_compressed']['current_bbw_percentile']})")
            print(f"   - 价格动能停滞: {analysis_result['details']['momentum_stalled']['status']} (历史百分位: {analysis_result['details']['momentum_stalled']['current_range_percentile']})")
        else:
            print(f"❌ [跳过] {stock} 未满足条件。 原因: {analysis_result.get('reason', '综合判断未通过')}")

    print("\n--- 扫描结束 ---")
    if fragile_stocks:
        print(f"\nสรุป: 共发现 {len(fragile_stocks)} 只符合条件的股票。")
    else:
        print("\nสรุป: 未发现符合“脆弱共识”阶段条件的股票。")