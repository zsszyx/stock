import sys
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append('d:\\stock\\v3')

# 导入我们刚创建的指标函数
import metric

# 创建测试数据
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=20, freq='D')
prices = 100 + np.cumsum(np.random.randn(20) * 0.5)

data = {
    'code': ['A'] * 20,
    'high': prices + np.random.rand(20) * 2,
    'low': prices - np.random.rand(20) * 2,
    'close': prices,
    'open': prices + np.random.randn(20) * 0.5
}

df = pd.DataFrame(data)
print("测试数据:")
print(df.head(10))

# 计算ADX指标
result = metric.adx(df, timeperiod=14)
print("\n计算ADX后的结果:")
print(result[['code', 'high', 'low', 'close', 'adx_14']].head(10))