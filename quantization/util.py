import pandas as pd
import numpy as np
from functools import reduce
import os 
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("util_main.INFO", encoding="utf-8")
    ]
)
def log_unhandled_exception(exc_type, exc_value, exc_traceback):
    logging.critical("程序异常退出", exc_info=(exc_type, exc_value, exc_traceback))
    # 如果是Ctrl+C，也继续调用默认处理
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = log_unhandled_exception
logger = logging.getLogger(__name__)
from sklearn.isotonic import spearmanr
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from init import with_db_connection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from tqdm import tqdm  # 用于显示进度条
from init import get_specific_stocks_latest_data
from quantization.factor import factor_names, mark_volume_support
import atexit
import traceback

def validate_df_coluns(df_list):
    """
    验证DataFrame列表中所有元素的列是否相等
    """
    if not df_list:
        return False  # 空列表被认为是无效的
    first_df = df_list[0]
    for df in df_list[1:]:
        if not df.columns.equals(first_df.columns):
            return False
    return True

def validate_time_line(df_list):
    """
    验证list中的df长度是否相等，并且date列的开头和首尾都相同
    """
    if not df_list:
        return False  # 空列表被认为是无效的
    first_df = df_list[0]
    for df in df_list[1:]:
        if not df.index.equals(first_df.index):
            logging.warning(f"时间线不一致: {first_df.index} vs {df.index}")
            return False
    return True

@with_db_connection
def get_industry_data(conn, cusor):
    sql = "select * from stock_industry"
    df = pd.read_sql(sql, conn)
    return df[['code', 'industry']]

def validate_industry_dummy(industry_cluster=None):
    if industry_cluster is None:
        df = get_industry_data()['industry']
    # df = get_industry_data()['industry']
    industry_nums = df.nunique()
    dummy = pd.get_dummies(df, drop_first=True)
    if dummy.isnull().values.any():
        logging.warning("Dummy变量中存在缺失值")
        return False
    if dummy.shape[1] != industry_nums - 1:
        logging.warning("Dummy变量的数量不正确")
        return False
    return True

def neutralize_features(df_list, cluster_dfs, cluster_names, code_list, feature_names, mask_list, date_col='date'):

    """
    通用中性化回归函数
    
    参数:
    df_list: 列表，每个元素是一个股票的DataFrame，包含日期和多个特征
    cluster_dfs: 列表，每个元素是一个分类的DataFrame，包含日期和多个特征
    feature_names: 列表，需要中性化的特征名称
    mask_list: 列表，与feature_names等长的布尔列表，标记哪些特征是次数特征(True)和非次数特征(False)
    date_col: 字符串，日期列的名称，默认为'date'
    
    返回:
    中性化后的df_list
    """
    
    # 验证输入参数
    if len(feature_names) != len(mask_list):
        raise ValueError("feature_names和mask_list长度必须相同")

    if not all(feature_name in df_list[0].columns for feature_name in feature_names):
        raise ValueError("feature_names中的名称超出了特征范围")
    
    assert validate_df_coluns(df_list), "df_list中的所有DataFrame列必须相同"
    # assert validate_time_line(df_list), "df_list中的所有DataFrame时间线必须相同"

    # 获取所有日期和股票代码
    all_dates = sorted(set.union(*[set(df[date_col]) for df in df_list]))
    stock_codes = code_list  # 使用提供的股票代码列表
    if len(stock_codes) != len(df_list):
        raise ValueError("stock_codes的长度必须与df_list相同")
    
    # 创建一个空的数据框来存储所有数据，便于横截面处理
    all_data = pd.DataFrame()
    
    # 合并所有股票的数据
    for i, df in enumerate(df_list):
        df_copy = df.copy()
        df_copy['code'] = stock_codes[i]
        all_data = pd.concat([all_data, df_copy], ignore_index=True)

    # 合并行业数据
    for cluster_df in cluster_dfs:
        all_data = all_data.merge(cluster_df, on=['code'], how='left')

    # 处理因子：预处理和中性化
    all_data = process_and_neutralize_factors(
        all_data, cluster_names, feature_names, mask_list, date_col

    )
    
    # # 将中性化后的值分拆回原始的df_list
    # for i, code in enumerate(code_list):
    #     stock_data = all_data[all_data['code'] == code].copy()
        
    #     # 添加中性化后的特征列到原始df
    #     for feature_name in feature_names:
    #         neutral_col = f"{feature_name}_neutral"
    #         df_list[i][f"{feature_name}_neutral"] = stock_data[neutral_col].values
    
    return all_data

def process_and_neutralize_factors(all_data, cluster_names, feature_names, mask_list, date_col):
    """
    处理因子：对次数特征进行百分比排名，并对所有特征进行中性化
    
    参数:
    all_data: 包含所有股票数据和分类数据的DataFrame
    feature_names: 需要处理的特征名称列表
    mask_list: 与feature_names等长的布尔列表，标记哪些特征是次数特征
    date_col: 日期列的名称
    
    返回:
    处理后的DataFrame
    """
    
    # 获取分类变量列名（假设分类变量都是数值型或已经转换为虚拟变量）
    # 这里我们假设除了日期、代码和特征列之外的都是分类变量
    cluster_cols = [col for col in cluster_names if col in all_data.columns]
    if len(cluster_cols) == 0:
        raise ValueError("没有找到分类变量列，请检查输入数据")
    
    # 对每个特征进行预处理和中性化
    for i, (feature_name, is_count_feature) in enumerate(zip(feature_names, mask_list)):
        # logging.info(f"处理特征 {i+1}/{len(feature_names)}: {feature_name} (类型: {'次数' if is_count_feature else '值'})")

        # 对次数特征进行百分比排名处理
        if is_count_feature:
            # 按日期分组计算百分比排名
            all_data[f"{feature_name}_preprocessed"] = all_data.groupby(date_col)[feature_name].transform(
                lambda x: x.rank(pct=True, method='average')
            )
        else:
            # 对值特征，直接使用原始值
            raise NotImplementedError("值特征的处理尚未实现")
        
        # 对每个日期进行横截面中性化回归
        neutralized_values = []
        
        # 对每个日期进行处理
        for date in sorted(all_data[date_col].unique()):
            logging.info(f"处理日期 {date} 的特征 {feature_name}")
            date_data = all_data[all_data[date_col] == date].copy()
            
            if len(date_data) < 3000:
                logging.warning(f"日期 {date} 的数据量: {len(date_data)}，跳过中性化")
                # neutralized_values.extend(date_data[f"{feature_name}_preprocessed"].values)
                continue
            # 
            y = date_data[f"{feature_name}_preprocessed"].values

            # 为分类变量创建虚拟变量（one-hot编码）
            # X_dummies = pd.DataFrame()
            # for cluster_col in cluster_cols:
            #     # 为每个分类变量创建虚拟变量，并丢弃基准组
            #     dummies = pd.get_dummies(date_data[cluster_col], prefix=cluster_col, drop_first=True)
        
            #     X_dummies = pd.concat([X_dummies, dummies], axis=1)
            X_dummies = pd.get_dummies(date_data[cluster_cols], drop_first=True)
            
            X = X_dummies.values.astype(float)
            
            # 添加常数项
            X = sm.add_constant(X)
            
            # 执行OLS回归
            logging.info(f"计算日期 {date}/长度{len(date_data)}，特征 {feature_name} 的回归模型")

            model = sm.OLS(y, X).fit()
            # 获取残差作为中性化后的值
            neutralized_values.extend(model.resid)
    
        # 将中性化后的值添加回数据框
        all_data[f"{feature_name}_neutral"] = neutralized_values
    
    return all_data

def test_neutralize_features():
    # 创建示例数据
    n_stocks = 10
    n_dates = 20
    n_features = 5

    # 创建df_list 和 code_list
    df_list = []
    code_list = []
    for i in range(n_stocks):
        code_list.append(f"stock_{i+1}")
        df = pd.DataFrame({
            "date": pd.date_range(start="2020-01-01", periods=n_dates),
            **{f"feature_{j}": np.random.randint(0, 100, n_dates) for j in range(1, n_features + 1)}
        })
        df_list.append(df)

    # 创建行业数据
    cluster_dfs = []
    for i in range(3):
        df = pd.DataFrame({
            "code": code_list,
            f"industry_{i+1}": np.random.randint(0, 3, n_stocks)
        })
        cluster_dfs.append(df)
    
    # 创建feature_names
    feature_names = [f"feature_{i}" for i in range(1, n_features + 1)]
    mask_list = [True] * n_features

    # 执行中性化
    df_list = neutralize_features(df_list, cluster_dfs, code_list, feature_names, mask_list, "date")

    return df_list

def evaluate_neutralized_factors(all_data, code_list, feature_names, date_col='date', forward_period=5):
    """
    评估中性化后因子的有效性
    
    参数:
    neutralized_dfs: 中性化后的df_list
    code_list: 股票代码列表
    feature_names: 特征名称列表（中性化后的列名，如['f0_neutral', 'f1_neutral']）
    date_col: 日期列名称
    forward_period: 预测未来多少期的收益
    """
    
    results = {}
    
    # 1. 合并所有数据以便分析
    # all_data = pd.DataFrame()
    # for i, df in enumerate(neutralized_dfs):
    #     df_copy = df.copy()
    #     df_copy['code'] = code_list[i]
    #     all_data = pd.concat([all_data, df_copy], ignore_index=True)
    
    # 2. 使用向量化操作计算未来收益率 (核心改动)
    # 首先确保数据按股票和日期排序
    all_data = all_data.sort_values(by=['code', date_col]).reset_index(drop=True)
    
    # 使用 groupby 和 shift 计算未来价格
    # shift(-forward_period) 会将未来N期的价格 "提前" 到当前行
    all_data['future_price'] = all_data.groupby('code')['close'].shift(-forward_period)
    
    # 计算未来收益率
    all_data['future_return'] = (all_data['future_price'] - all_data['close']) / all_data['close']
    
    # 3. 评估每个因子
    for feature_name in feature_names:
        print(f"\n评估因子: {feature_name}")
        
        # 计算IC（信息系数）
        ic_values = []
        valid_dates = []
        
        # 按日期分组，计算每个截面的IC值
        # 使用 apply 比循环更高效
        def calculate_ic(group):
            # 过滤掉没有未来收益的数据
            group = group.dropna(subset=[feature_name, 'future_return'])
            if len(group) < 10:
                return None
            
            # 计算Spearman秩相关系数
            ic, _ = spearmanr(group[feature_name], group['future_return'])
            return ic

        # 使用 groupby().apply() 计算每个日期的IC
        ic_series = all_data.groupby(date_col).apply(calculate_ic).dropna()
        
        # 4. 计算IC统计量
        if not ic_series.empty:
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ir = ic_mean / ic_std if ic_std != 0 else 0
            
            results[feature_name] = {
                'IC_mean': ic_mean,
                'IC_std': ic_std,
                'IR': ir,
                'IC_series': ic_series
            }
            
            print(f"  IC均值: {ic_mean:.4f}, IC标准差: {ic_std:.4f}, IR: {ir:.4f}")
        else:
            print(f"  无法计算因子 {feature_name} 的IC值 (数据不足或全为NaN)")
    
    return results

def test_evaluate_neutralized_factors():
    # 创建示例数据
    # 创建示例数据
    n_stocks = 10
    n_dates = 20
    n_features = 5

    # 创建df_list 和 code_list
    df_list = []
    code_list = []
    for i in range(n_stocks):
        code_list.append(f"stock_{i+1}")
        df = pd.DataFrame({
            "date": pd.date_range(start="2020-01-01", periods=n_dates),
            "close": np.random.uniform(1, 200, n_dates),
            **{f"feature_{j}": np.random.randint(0, 100, n_dates) for j in range(1, n_features + 1)}
        })
        df_list.append(df)

    # 创建行业数据
    cluster_dfs = []
    for i in range(3):
        df = pd.DataFrame({
            "code": code_list,
            f"industry_{i+1}": np.random.randint(0, 3, n_stocks)
        })
        cluster_dfs.append(df)
    
    # 创建feature_names
    feature_names = [f"feature_{i}" for i in range(1, n_features + 1)]
    mask_list = [True] * n_features

    # 执行中性化
    df_list = neutralize_features(df_list, cluster_dfs, code_list, feature_names, mask_list, "date")
    results = evaluate_neutralized_factors(
        df_list, code_list, [f"{name}_neutral" for name in feature_names], date_col='date', forward_period=5
    )
    return results

def alignment_time_line(df_list):
    """
    对齐时间线
    """
    # 获取所有DataFrame的日期索引的并集
    logging.info(f"原始时间线: {df_list[0]['date']}")

    all_dates = set()
    for df in df_list:
        all_dates.update(df['date'])
    all_dates = sorted(all_dates)

    # 统计每个日期在多少个df中出现
    date_counts = {}
    for date in all_dates:
        count = sum(date in set(df['date']) for df in df_list)
        date_counts[date] = count

    # 选择所有df都包含的最大连续日期区间
    # 先找所有df都包含的日期
    full_dates = [date for date in all_dates if date_counts[date] == len(df_list)]
    if not full_dates:
        raise ValueError("没有所有DataFrame都包含的日期，无法对齐")

    # 找最大连续区间
    max_seq = []
    temp_seq = []
    for i, date in enumerate(full_dates):
        if not temp_seq:
            temp_seq = [date]
        else:
            prev_date = temp_seq[-1]
            if (pd.to_datetime(date) - pd.to_datetime(prev_date)).days == 1:
                temp_seq.append(date)
            else:
                if len(temp_seq) > len(max_seq):
                    max_seq = temp_seq
                temp_seq = [date]
    if len(temp_seq) > len(max_seq):
        max_seq = temp_seq

    # 只保留最大连续区间的日期
    aligned_dates = set(max_seq)
    logging.info(f"对齐后的时间线: {sorted(aligned_dates)}")
    aligned_df_list = []
    for df in df_list:
        aligned_df = df[df['date'].isin(aligned_dates)].copy()
        aligned_df = aligned_df.sort_values('date').reset_index(drop=True)
        aligned_df_list.append(aligned_df)
    return aligned_df_list

def main():
    stock_data = get_specific_stocks_latest_data(length=22)
    code_list = []
    df_list = []
    for num, i in enumerate(stock_data.values()):
        code_list.append(i['code'])
        logging.info(f"处理股票 {i['code']},({num}/{len(stock_data)})")
        df = mark_volume_support(i['data'])
        df_list.append(df)
    # df_list = alignment_time_line(df_list)
    cluster_dfs = [get_industry_data()]
    cluster_names = ['industry']
    feature_names = [factor_names[0]]
    mask_list = [True] * len(feature_names)
    all_data = neutralize_features(df_list, cluster_dfs, cluster_names, code_list, feature_names, mask_list, "date")
    results = evaluate_neutralized_factors(
        all_data, code_list, [f"{name}_neutral" for name in feature_names], date_col='date', forward_period=5
    )
    return results




# 示例使用方式
if __name__ == "__main__":
#    print(test_neutralize_features())
#    test_evaluate_neutralized_factors()
#    main()
   validate_industry_dummy()

