import pandas as pd
import numpy as np
import functools
import os
import sys
import datetime
import sqlite3
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import logging
from sklearn.isotonic import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from tqdm import tqdm
from factors.factor import factor_names, factor_dict, mask_dict, get_factor_merge_table

def get_logfile_with_time(log_dir="logs", prefix="util_main"):
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"{prefix}.{now}.log")

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stock_data.db')
# DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_data.db')

def with_db_connection(func):
    """数据库连接的装饰器，自动处理连接的创建、提交和关闭"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        logging.info(f"数据库连接已建立: {DB_PATH}")
        result = func(conn=conn, cursor=cursor, *args, **kwargs)
        conn.commit()
        logging.info("数据库操作已提交")
        cursor.close()
        logging.info("数据库连接已关闭")
        return result
    return wrapper

class FactorTask:
    def __init__(self, all_data, cluster_names=None, feature_names=None, mask_list=None, date_col='date', forward_period=5, log_file=None):
        self.all_data = all_data
        self.cluster_names = cluster_names if cluster_names is not None else ['industry']
        self.feature_names = feature_names if feature_names is not None else factor_names
        self.mask_list = mask_list if mask_list is not None else [mask_dict[i] for i in self.feature_names]
        self.date_col = date_col
        self.forward_period = forward_period
        # 日志文件名带时间戳
        if log_file is None:
            log_file = get_logfile_with_time()
        self.log_file = log_file
        self.logger = logging.getLogger(f"FactorTask_{id(self)}")
        self.logger.setLevel(logging.INFO)
        # 防止重复添加handler
        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_file, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def process_and_neutralize_factors(self):
        all_data = self.all_data.copy()
        cluster_names = self.cluster_names
        feature_names = self.feature_names
        mask_list = self.mask_list
        date_col = self.date_col
        cluster_cols = [col for col in cluster_names if col in all_data.columns]
        if len(cluster_cols) == 0:
            raise ValueError("没有找到分类变量列，请检查输入数据")
        xdummy = pd.get_dummies(all_data[cluster_cols], drop_first=True)
        xdummy.index = all_data.index
        log_market_cap = np.log1p(all_data['market_value'])
        log_market_cap.index = all_data.index
        x_values = pd.concat([log_market_cap, xdummy], axis=1)
        x_values = sm.add_constant(x_values)
        for i, (feature_name, is_count_feature) in enumerate(zip(feature_names, mask_list)):
            self.logger.info(f"处理特征 {i+1}/{len(feature_names)}: {feature_name} (类型: {'次数' if is_count_feature else '值'})")
            preprocessed_col = f"{feature_name}_preprocessed"
            if is_count_feature:
                all_data[preprocessed_col] = all_data.groupby(date_col)[feature_name].transform(
                    lambda x: x.rank(pct=True, method='average')
                )
            else:
                all_data[preprocessed_col] = (all_data[feature_name] - all_data[feature_name].mean()) / all_data[feature_name].std()
            neutralized_col = f"{feature_name}_neutral"
            all_data[neutralized_col] = np.nan
            for date in sorted(all_data[date_col].unique()):
                self.logger.info(f"处理日期 {date} 的特征 {feature_name}")
                date_mask = all_data[date_col] == date
                date_data = all_data[date_mask]
                y = date_data[preprocessed_col].values
                valid_mask = pd.notna(y)
                if valid_mask.sum() < 30:
                    self.logger.warning(f"日期 {date} 的有效数据量: {valid_mask.sum()}，跳过中性化")
                    continue
                temp_x_values = x_values.loc[date_data.index[valid_mask]]
                model = sm.OLS(y[valid_mask], temp_x_values.astype(float)).fit()
                date_indices = all_data.index[date_mask]
                valid_indices = date_indices[valid_mask]
                all_data.loc[valid_indices, neutralized_col] = model.resid
            all_data = all_data.drop(columns=[preprocessed_col, feature_name])
        self.all_data = all_data
        return all_data

    def evaluate_neutralized_factors(self):
        all_data = self.all_data.copy()
        feature_names = [f"{name}_neutral" for name in self.feature_names]
        date_col = self.date_col
        forward_period = self.forward_period
        results = {}
        all_data = all_data.sort_values(by=['code', date_col]).reset_index(drop=True)
        all_data['future_price'] = all_data.groupby('code')['close'].shift(-forward_period)
        all_data['future_return'] = (all_data['future_price'] - all_data['close']) / all_data['close']
        for feature_name in feature_names:
            self.logger.info(f"评估因子: {feature_name}")

            def calculate_ic(group):
                valid_group = group.dropna(subset=[feature_name, 'future_return'])
                if len(valid_group) < 10:
                    self.logger.info(f"在日期 {group[date_col].iloc[0]}, 因子 '{feature_name}' 或收益率是常数，无法计算IC")
                    return np.nan
                factor_values = valid_group[feature_name]
                return_values = valid_group['future_return']
                ic, _ = spearmanr(factor_values, return_values)
                return ic
            
            ic_series = all_data.groupby(date_col).apply(calculate_ic).dropna()

            if not ic_series.empty:
                ic_mean = ic_series.mean()  
                ic_std = ic_series.std()
                ir = ic_mean / ic_std
                results[feature_name] = {
                    'IC_mean': ic_mean,
                    'IC_std': ic_std,
                    'IR': ir,
                    'IC_samples': len(ic_series),
                }
                self.logger.info(f"因子{feature_name}, IC均值: {ic_mean:.4f}, IC标准差: {ic_std:.4f}, IR: {ir:.4f}, 样本数: {len(ic_series)}")
            else:
                self.logger.warning(f'无法计算因子 {feature_name} 的IC值 (数据不足或全为NaN)')
        return results, all_data
    
    @with_db_connection
    def run(self, conn, cursor, save_data_path=None):
        logging.info(f"在分类变量 {self.cluster_names+['market_cap']} 下处理中性化特征: {self.feature_names}")
        self.process_and_neutralize_factors()
        results, all_data = self.evaluate_neutralized_factors()
        if save_data_path:
            all_data.to_sql(save_data_path, if_exists='replace', index=False, con=conn)
        return results, all_data

class CorrelationTask:
    """
    计算并分析中性化因子之间的相关性。
    """
    def __init__(self, table_name='neutralized_factors_meta_data'):
        self.table_name = table_name
        self.logger = logging.getLogger(f"CorrelationTask_{id(self)}")
        # 配置日志，如果还没有配置的话
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    @with_db_connection
    def run(self, conn, cursor):
        """
        执行相关性分析。

        :param conn: 数据库连接对象
        :param cursor: 数据库游标对象
        :return: 一个DataFrame，包含因子对和它们的相关性，按相关性降序排列。
        """
        self.logger.info(f"开始从表 '{self.table_name}' 中读取数据进行相关性分析。")
        try:
            df = pd.read_sql(f'SELECT * FROM {self.table_name}', conn)
            self.logger.info(f"成功读取 {len(df)} 条数据。")
        except Exception as e:
            self.logger.error(f"无法从数据库读取表 '{self.table_name}'。错误: {e}")
            self.logger.error("请确保 'FactorTask' 已成功运行并生成了此表。")
            return None

        # 筛选出所有中性化因子列
        neutral_factors = [col for col in df.columns if col.endswith('_neutral')]
        if len(neutral_factors) < 2:
            self.logger.warning("找到的中性化因子少于2个，无法进行相关性分析。")
            return None
        
        self.logger.info(f"找到 {len(neutral_factors)} 个中性化因子进行分析。")

        # 计算相关性矩阵
        correlation_matrix = df[neutral_factors].corr().abs()

        # 提取上三角矩阵（k=1确保不包括对角线），避免重复对 (A,B) 和 (B,A) 以及自身对 (A,A)
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

        # 将矩阵转换为长格式的列表
        correlation_pairs = upper_triangle.stack().reset_index()
        correlation_pairs.columns = ['Factor_1', 'Factor_2', 'Correlation']

        # 按相关性降序排序
        sorted_pairs = correlation_pairs.sort_values(by='Correlation', ascending=False).reset_index(drop=True)
        
        # 移除因子名称中的 '_neutral' 后缀以提高可读性
        sorted_pairs['Factor_1'] = sorted_pairs['Factor_1'].str.replace('_neutral', '')
        sorted_pairs['Factor_2'] = sorted_pairs['Factor_2'].str.replace('_neutral', '')
        
        self.logger.info("相关性分析完成。")
        return sorted_pairs

if __name__ == "__main__":
    # 配置基本日志记录
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # # 示例用法
    # df = get_factor_merge_table()
    # # 日志文件名带时间戳
    # log_file = get_logfile_with_time()
    # task = FactorTask(all_data=df, cluster_names=['industry'], feature_names=list(factor_dict.keys()), mask_list=[False]*len(factor_dict), log_file=log_file)
    # results, all_data = task.run(save_data_path='neutralized_factors_meta_data')
    # print("因子评估结果:")
    # print(results)

    # 新增：运行相关性分析任务
    correlation_task = CorrelationTask(table_name='neutralized_factors_meta_data')
    correlation_results = correlation_task.run()
    if correlation_results is not None:
        print("\n中性化因子相关性分析结果 (Top 10):")
        print(correlation_results.head(10))

