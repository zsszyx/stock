import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score

# 添加项目路径
sys.path.append('d:\\stock\\v3')

# 从train.py导入相关函数
from train import get_stock_merge_industry_table, adtm, vrsi, volume_ratio, create_features_with_lags, prepare_training_data

def test_model_parameters():
    """
    测试调整后的模型参数效果
    """
    print("开始测试模型参数...")
    
    # 加载少量数据进行测试
    df = get_stock_merge_industry_table(length=100)
    df = adtm(df)
    df = vrsi(df)
    df = volume_ratio(df)
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    # 定义特征列
    feature_columns = ['atr14', 'pctChg', 'turn', 'adtm_15', 'adtmma_15', 'vr_5', 'vrsi_8']
    
    # 创建滞后特征
    df = create_features_with_lags(df, feature_columns, lag_days=5)
    
    # 删除包含NaN的行
    df = df.dropna()
    
    print(f"处理后的数据有 {len(df)} 条记录")
    
    # 构建特征列名
    feature_cols = []
    for col in feature_columns:
        for lag in range(5):
            feature_cols.append(f'{col}_lag_{lag}')
    
    # 简单地创建一个标签（这里只是示例）
    df['label'] = (df['pctChg'] > 0).astype(int)
    
    # 提取特征和标签
    X = df[feature_cols]
    y = df['label']
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"标签分布: {y.value_counts()}")
    
    # 使用调整后的参数创建模型
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
    
    # 简单划分训练集和测试集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 训练模型
    print("开始训练模型...")
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"测试准确率: {accuracy:.4f}")
    print(f"测试F1分数: {f1:.4f}")
    
    # 显示特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n前10个最重要的特征:")
    print(feature_importance.head(10))
    
    return model, accuracy, f1

if __name__ == "__main__":
    test_model_parameters()