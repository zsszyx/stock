import sys
import pandas as pd
import numpy as np

# 添加模块搜索路径
sys.path.append('d:\\stock\\v3')

import metric

# 创建测试数据
df = pd.DataFrame({
    'open': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'high': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    'low': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    'code': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A']
})

# 计算ADTM
result = metric.adtm(df, timeperiod=5)
print(result[['open', 'high', 'low', 'adtm_5', 'adtmma_5']].tail())