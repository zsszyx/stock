import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb

# 添加项目路径
sys.path.append('d:\\stock\\v3')

# 导入必要的函数
from data import get_stock_merge_industry_table
from metric import adtm, vrsi, volume_ratio, adx, price_change

# 模拟create_features_with_lags函数
def create_features_with_lags(df, feature_columns, lag_days):
    """创建滞后特征"""
    for col in feature_columns:
        for lag in range(1, lag_days + 1):  # 从1开始，避免使用当天数据
            df[f'{col}_lag_{lag}'] = df.groupby('code')[col].shift(lag)
    return df

# 模拟prepare_training_data函数
def prepare_training_data(df, feature_columns, lag_days=15):
    """准备训练数据"""
    # 创建滞后特征
    df = create_features_with_lags(df, feature_columns, lag_days)
    
    # 删除包含NaN的行（由于滞后特征产生）
    df = df.dropna()
    
    # 构建特征列名
    feature_cols = []
    for col in feature_columns:
        for lag in range(1, lag_days + 1):  # 从1开始，避免使用当天数据
            feature_cols.append(f'{col}_lag_{lag}')
    
    # 创建一个简单的标签（这里我们使用pctChg作为示例）
    df['label'] = (df['pctChg'] > 0).astype(int)
    
    # 提取特征和标签
    X = df[feature_cols]
    y = df['label']
    
    return X, y, df

def main():
    print("开始测试当前模型参数...")
    
    # 定义特征列
    feature_columns = ['atr14', 'pctChg', 'turn', 'adtm_15', 
                       'adtmma_15', 'vr_5', 'vrsi_8', 'adx_14', 
                       'adxr_14', 'open_pct', 'high_pct', 'low_pct', 
                       'close_pct']
    
    # 创建一些模拟数据进行测试
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    codes = ['A', 'B', 'C']
    
    data = []
    for code in codes:
        for date in dates:
            data.append({
                'code': code,
                'date': date,
                'open': np.random.rand() * 100,
                'high': np.random.rand() * 100 + 5,
                'low': np.random.rand() * 100 - 5,
                'close': np.random.rand() * 100,
                'volume': np.random.randint(1000, 10000),
                'turn': np.random.rand() * 10,
                'pctChg': (np.random.rand() - 0.5) * 10
            })
    
    df = pd.DataFrame(data)
    df = df.set_index(['code', 'date'])
    
    # 计算技术指标
    df = adtm(df)
    df = vrsi(df)
    df = volume_ratio(df)
    df = adx(df)
    df = price_change(df)
    
    # 准备训练数据
    X, y, processed_df = prepare_training_data(df, feature_columns, lag_days=15)
    
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    print(f"标签分布: {y.value_counts().to_dict()}")
    
    # 使用当前参数创建模型
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42)
    
    # 训练模型
    print("开始训练模型...")
    model.fit(X, y)
    
    # 进行预测
    y_pred = model.predict(X)
    
    # 计算评估指标
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    
    print(f"准确率: {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("模型参数测试完成!")

if __name__ == "__main__":
    main()