import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import logging
from datetime import timedelta
from prepare import get_stock_merge_industry_table
from metric import adx, atr, adtm, price_change, rsi, volume_ratio, vrsi
import os

# 配置日志
# 创建logs目录（如果不存在）
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "training.log"), encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


def calculate_dynamic_barriers(df, atr_period=14, upper_multiplier=2.0, lower_multiplier=0.5, time_barrier=5):
    """
    计算动态障碍位
    
    参数:
    df: 包含股票数据的DataFrame，必须包含'close'列
    atr_period: ATR计算周期
    upper_multiplier: 上轨乘数
    lower_multiplier: 下轨乘数
    time_barrier: 时间障碍（交易日）
    
    返回:
    添加了障碍位相关列的DataFrame
    """
    # 保存原始索引
    original_index = df.index
    
    # 计算ATR
    df = atr(df, timeperiod=atr_period)
    atr_col = f'atr{atr_period}'
    
    # 计算动态障碍位
    df['upper_barrier'] = df['close'] + (upper_multiplier * df[atr_col])
    df['lower_barrier'] = df['close'] - (lower_multiplier * df[atr_col])
    df['time_barrier'] = time_barrier  # 时间障碍天数
    
    # 恢复原始索引，避免索引重复
    # df.index = original_index
    
    return df


def create_labels_with_barriers(df, forward_days=10):
    """
    根据障碍位创建标签
    
    参数:
    df: 包含价格和障碍位数据的DataFrame
    forward_days: 向前看的天数
    
    返回:
    带有标签的DataFrame
    """
    # 保存原始索引
    original_index = df.index
    
    # 计算未来N天内的最高价和最低价
    df['future_high'] = df['high'].rolling(window=forward_days).max().shift(-forward_days+1)
    df['future_low'] = df['low'].rolling(window=forward_days).min().shift(-forward_days+1)
    
    # 初始化标签列
    df['label'] = 0  # 0: 持平, 1: 上涨, -1: 下跌
    
    # 设置标签逻辑
    # 1: 价格上涨突破上轨
    # -1: 价格下跌突破下轨
    # 0: 在时间内未突破任一轨道
    
    # 检查是否突破上轨
    upper_breakthrough = (df['future_high'] >= df['upper_barrier'])
    # 检查是否突破下轨
    lower_breakthrough = (df['future_low'] <= df['lower_barrier'])
    
    # 设置标签
    df.loc[upper_breakthrough & ~lower_breakthrough, 'label'] = 1  # 上涨
    df.loc[lower_breakthrough & ~upper_breakthrough, 'label'] = -1  # 下跌
    # 默认为0（持平），表示在时间窗口内未突破任一轨道
    
    # 恢复原始索引，避免索引重复
    # df.index = original_index
    
    return df


def create_features_with_lags(df, feature_columns, lag_days=5):
    """
    创建滞后特征
    
    参数:
    df: DataFrame
    feature_columns: 特征列名列表
    lag_days: 滞后天数
    
    返回:
    带有滞后特征的DataFrame
    """
    # 检查df是否有MultiIndex且包含'code'层级
    if isinstance(df.index, pd.MultiIndex) and 'code' in df.index.names:
        # 按股票代码分组创建滞后特征，避免不同股票数据串用
        grouped = df.groupby(level='code')
        for col in feature_columns:
            for lag in range(lag_days):
                df[f'{col}_lag_{lag}'] = grouped[col].shift(lag)
    else:
        raise ValueError("DataFrame必须是MultiIndex，包含'code'层级")
    return df


def prepare_training_data(df, feature_columns, label_column='label', lag_days=5):
    """
    准备训练数据
    
    参数:
    df: 包含特征和标签的DataFrame
    feature_columns: 特征列名列表
    label_column: 标签列名
    lag_days: 滞后天数
    
    返回:
    特征矩阵X和标签向量y
    """
    # 创建滞后特征
    df = create_features_with_lags(df, feature_columns, lag_days)
    
    # 删除包含NaN的行（由于滞后特征产生）
    df = df.dropna()
    
    # 构建特征列名
    feature_cols = []
    for col in feature_columns:
        for lag in range(lag_days):
            feature_cols.append(f'{col}_lag_{lag}')
    
    # 提取特征和标签
    X = df[feature_cols]
    y = df[label_column]
    
    return X, y, df


def rolling_window_train(df, n_days=30, m_days=10, lag_days=3, train_repeats=1):
    """
    滚动时间窗口训练
    
    参数:
    df: 包含所有股票数据的DataFrame
    n_days: 训练数据的天数
    m_days: 预测的时间间隔（天）
    lag_days: 特征滞后期数
    train_repeats: 每个窗口上训练模型的次数，用于多次训练以提高模型性能
    
    返回:
    训练结果和模型列表
    """

    model = lgb.LGBMClassifier(
        n_estimators=200,           # 增加树的数量以提高模型性能
        learning_rate=0.05,         # 降低学习率以获得更精细的学习
        max_depth=8,                # 增加最大深度以捕获更复杂的模式
        num_leaves=63,              # 设置叶子节点数，通常为2^max_depth-1
        min_child_samples=20,       # 增加子节点最小样本数以防止过拟合
        subsample=0.8,              # 使用80%的样本进行训练以增加泛化能力
        colsample_bytree=0.8,       # 使用80%的特征进行训练以增加泛化能力
        reg_alpha=0.1,              # L1正则化防止过拟合
        reg_lambda=0.1,             # L2正则化防止过拟合
        random_state=42)

    # 确保索引是日期类型
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame必须是MultiIndex，包含code和date")
    
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
        
        logger.info(f"训练窗口: {train_start_date} 到 {train_end_date}")
        logger.info(f"预测窗口: {predict_start_date} 到 {predict_end_date}")
        
        # 获取训练数据
        # 使用布尔索引正确筛选日期范围内的数据
        train_mask = (df.index.get_level_values('date') >= train_start_date) & (df.index.get_level_values('date') <= train_end_date)
        train_data = df[train_mask]
        
        # 准备训练数据
        X_train, y_train, _ = prepare_training_data(train_data, feature_columns, lag_days=lag_days)
        
        if len(X_train) == 0:
            logger.warning("训练数据为空，跳过此窗口")
            start_idx += m_days
            continue
        
        # 训练模型（重复训练指定次数）
        logger.info(f"训练模型，使用 {len(X_train)} 条训练样本，重复训练 {train_repeats} 次")
        # 显示标签分布情况
        label_counts = y_train.value_counts().sort_index()
        label_distribution = {label: count for label, count in label_counts.items()}
        logger.info(f"标签分布: {label_distribution}")
            
        # 重复训练模型
        for repeat in range(train_repeats):
            if repeat == 0:
                # 第一次训练
                model.fit(X_train, y_train)
            else:
                # 后续训练使用增量学习
                model.fit(X_train, y_train, init_model=model)
        
        # 获取预测数据
        # 使用布尔索引正确筛选日期范围内的数据
        predict_mask = (df.index.get_level_values('date') >= predict_start_date) & (df.index.get_level_values('date') <= predict_end_date)
        predict_data = df[predict_mask]
        
        # 准备预测数据
        X_predict, y_predict, processed_predict_data = prepare_training_data(predict_data, feature_columns, lag_days=lag_days)
        
        if len(X_predict) == 0:
            logger.warning("预测数据为空，跳过此窗口")
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
        # 只考虑标签为1的样本
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
        
        logger.info(f"窗口结果 - 准确率: {accuracy:.4f}, 标签1准确率: {label_1_accuracy:.4f}, F1分数: {f1:.4f}")
        logger.info(f"==================================================")
        
        # 移动到下一个窗口
        start_idx += m_days
    
    return model, results


def main():
    """
    主函数：执行完整的训练流程
    """
    logger.info("开始加载数据...")
    
    # 加载数据
    df = get_stock_merge_industry_table(length=400)  # 获取足够多的数据用于训练
    df = adtm(df)
    df = vrsi(df)
    df = volume_ratio(df)
    df = adx(df)
    df = price_change(df)
    df = rsi(df)
    
    logger.info(f"数据加载完成，共 {len(df)} 条记录")
    
    # 执行滚动窗口训练
    logger.info("开始滚动窗口训练...")
    models, results = rolling_window_train(df, n_days=40, m_days=40, lag_days=15, train_repeats=2)
    
    # 输出结果统计
    if results:
        avg_accuracy = np.mean([r['accuracy'] for r in results])
        avg_label_1_accuracy = np.mean([r['label_1_accuracy'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        
        logger.info("=== 训练结果汇总 ===")
        logger.info(f"平均准确率: {avg_accuracy:.4f}")
        logger.info(f"平均标签1准确率: {avg_label_1_accuracy:.4f}")
        logger.info(f"平均F1分数: {avg_f1:.4f}")
        
        # 输出详细结果
        for i, result in enumerate(results):
            logger.info(f"窗口 {i+1}: "
                       f"训练期 {result['train_start']} 到 {result['train_end']}, "
                       f"预测期 {result['predict_start']} 到 {result['predict_end']}, "
                       f"准确率 {result['accuracy']:.4f}, "
                       f"标签1准确率 {result['label_1_accuracy']:.4f}, "
                       f"F1分数 {result['f1']:.4f}")
    else:
        logger.warning("没有生成任何训练结果")


if __name__ == "__main__":
    main()