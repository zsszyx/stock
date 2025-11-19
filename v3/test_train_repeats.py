import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb

# 添加项目路径
sys.path.append('d:\\stock\\v3')

# 导入必要的函数
from data import get_stock_merge_industry_table
from metric import adtm, vrsi, volume_ratio, adx, price_change, rsi

# 模拟create_features_with_lags函数
def create_features_with_lags(df, feature_columns, lag_days):
    """创建滞后特征"""
    for col in feature_columns:
        for lag in range(1, lag_days + 1):  # 从1开始，避免使用当天数据
            df[f'{col}_lag_{lag}'] = df.groupby('code')[col].shift(lag)
    return df

# 模拟prepare_training_data函数
def prepare_training_data(df, feature_columns, lag_days=15, label_column='label'):
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
    y = df[label_column] if label_column in df.columns else df['label']
    
    return X, y, df

# 模拟calculate_dynamic_barriers函数
def calculate_dynamic_barriers(df):
    """计算动态障碍位"""
    # 简化实现，实际应该根据ATR计算上下障碍位
    df['upper_barrier'] = df['close'] * 1.05
    df['lower_barrier'] = df['close'] * 0.95
    return df

# 模拟create_labels_with_barriers函数
def create_labels_with_barriers(df):
    """根据障碍位创建标签"""
    # 简化实现
    df['label'] = (df['pctChg'] > 0).astype(int)
    return df

def rolling_window_train(df, n_days=30, m_days=10, lag_days=3, train_repeats=1):
    """
    滚动时间窗口训练
    
    参数:
    df: 包含所有股票数据的DataFrame
    n_days: 训练数据的天数
    m_days: 预测的时间间隔（天）
    lag_days: 特征滞后期数
    train_repeats: 每个窗口上训练模型的次数
    
    返回:
    训练结果和模型列表
    """
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import numpy as np
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42)

    # 确保索引是日期类型
    if not isinstance(df.index, pd.MultiIndex):
        df = df.set_index(['code', 'date'])
    
    # 获取所有日期
    all_dates = sorted(df.index.get_level_values('date').unique())
    
    # 定义特征列
    feature_columns = ['atr14', 'pctChg', 'turn', 'adtm_15', 
                       'adtmma_15', 'vr_5', 'vrsi_8', 'adx_14', 
                       'adxr_14', 'open_pct', 'high_pct', 'low_pct', 
                       'close_pct', 'rsi_14']
    
    # 计ATR和障碍位
    df = df.groupby('code', group_keys=False).apply(lambda x: calculate_dynamic_barriers(x))
    
    # 创建标签
    df = df.groupby('code', group_keys=False).apply(lambda x: create_labels_with_barriers(x))
    
    results = []
    
    # 开始滚动窗口训练
    start_idx = n_days
    while start_idx + m_days <= len(all_dates):
        # 定义训练和预测时间窗口
        train_start_date = all_dates[start_idx - n_days]
        train_end_date = all_dates[start_idx - 1]  # 训练结束日期是开始预测的前一天
        predict_start_date = all_dates[start_idx]
        predict_end_date = all_dates[start_idx + m_days - 1]
        
        print(f"训练窗口: {train_start_date} 到 {train_end_date}")
        print(f"预测窗口: {predict_start_date} 到 {predict_end_date}")
        
        # 获取训练数据
        train_mask = (df.index.get_level_values('date') >= train_start_date) & (df.index.get_level_values('date') <= train_end_date)
        train_data = df[train_mask]
        
        # 准备训练数据
        X_train, y_train, _ = prepare_training_data(train_data, feature_columns, lag_days=lag_days)
        
        if len(X_train) == 0:
            print("训练数据为空，跳过此窗口")
            start_idx += m_days
            continue
        
        # 训练模型（重复训练指定次数）
        print(f"训练模型，使用 {len(X_train)} 条训练样本，重复训练 {train_repeats} 次")
        # 显示标签分布情况
        label_counts = y_train.value_counts().sort_index()
        label_distribution = {label: count for label, count in label_counts.items()}
        print(f"标签分布: {label_distribution}")
            
        # 重复训练模型
        for repeat in range(train_repeats):
            if repeat == 0:
                # 第一次训练
                model.fit(X_train, y_train)
            else:
                # 后续训练使用增量学习
                model.fit(X_train, y_train, init_model=model)
        
        # 获取预测数据
        predict_mask = (df.index.get_level_values('date') >= predict_start_date) & (df.index.get_level_values('date') <= predict_end_date)
        predict_data = df[predict_mask]
        
        # 准备预测数据
        X_predict, y_predict, processed_predict_data = prepare_training_data(predict_data, feature_columns, lag_days=lag_days)
        
        if len(X_predict) == 0:
            print("预测数据为空，跳过此窗口")
            start_idx += m_days
            continue
        
        # 进行预测
        y_pred = model.predict(X_predict)
        
        # 计算评估指标
        accuracy = accuracy_score(y_predict, y_pred)
        precision = precision_score(y_predict, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_predict, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_predict, y_pred, average='weighted', zero_division=0)
        
        # 计算标签1（上涨撞线）的准确率
        label_1_mask = (y_predict == 1)
        if np.sum(label_1_mask) > 0:
            label_1_accuracy = accuracy_score(y_predict[label_1_mask], y_pred[label_1_mask])
        else:
            label_1_accuracy = 0.0
        
        result = {
            'train_start': train_start_date,
            'train_end': train_end_date,
            'predict_start': predict_start_date,
            'predict_end': predict_end_date,
            'accuracy': accuracy,
            'label_1_accuracy': label_1_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'samples': len(X_predict)
        }
        results.append(result)
        
        print(f"窗口结果 - 准确率: {accuracy:.4f}, 标签1准确率: {label_1_accuracy:.4f}, F1分数: {f1:.4f}")
        print("=" * 50)
        
        # 移动到下一个窗口
        start_idx += m_days
    
    return model, results

def main():
    print("开始测试train_repeats超参数功能...")
    
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
    df = rsi(df)
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    # 测试不同的train_repeats值
    for train_repeats in [1, 3, 5]:
        print(f"\n测试 train_repeats = {train_repeats}")
        models, results = rolling_window_train(df, n_days=40, m_days=20, lag_days=5, train_repeats=train_repeats)
        
        if results:
            avg_accuracy = np.mean([r['accuracy'] for r in results])
            avg_f1 = np.mean([r['f1'] for r in results])
            print(f"平均准确率: {avg_accuracy:.4f}")
            print(f"平均F1分数: {avg_f1:.4f}")
        else:
            print("没有生成任何训练结果")

if __name__ == "__main__":
    main()