import sys
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append('d:\\stock\\v3')

# 导入我们刚创建的指标函数
import metric

# 创建测试数据
data = {
    'code': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
    'volume': [1000, 1200, 800, 1500, 2000, 1800, 2200, 2500, 3000, 2800, 3200, 3500, 3300, 3600, 4000]
}

df = pd.DataFrame(data)
print("测试数据:")
print(df)

# 计算VRSI指标
result = metric.vrsi(df, timeperiod=14)
print("\n计算VRSI后的结果:")
print(result[['code', 'volume', 'vrsi_14']])