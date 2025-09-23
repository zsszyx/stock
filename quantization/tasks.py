import pandas as pd
import numpy as np
import functools
import os
import sys
import datetime
import sqlite3
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import logging
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from tqdm import tqdm
from factors.factor import factor_names, factor_dict, mask_dict, get_factor_merge_table

def setup_logger(name, log_file=None, level=logging.INFO):
    """Function to set up a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 防止日志消息传递到父记录器

    if not logger.handlers:
        handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

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
        self.cluster_names = cluster_names 
        self.feature_names = feature_names 
        self.mask_list = mask_list 
        self.date_col = date_col
        self.forward_period = forward_period
        self.log_file = log_file
        self.logger = setup_logger(f"FactorTask_{id(self)}", self.log_file)

    def process_and_neutralize_factors(self):
        all_data = self.all_data.copy()
        cluster_names = self.cluster_names
        feature_names = self.feature_names
        mask_list = self.mask_list
        date_col = self.date_col
        cluster_cols = [col for col in cluster_names if col in all_data.columns]

        xdummy = pd.get_dummies(all_data[cluster_cols], drop_first=True)
        xdummy.index = all_data.index

        log_market_cap = np.log1p(all_data['market_value'])
        log_market_cap.index = all_data.index

        x_values = pd.concat([log_market_cap, xdummy], axis=1)
        x_values = sm.add_constant(x_values)
        
        def _neutralize_single_date(group, feature_col, x_vals):
            """对单个交易日的数据进行因子中性化"""
            y = group[feature_col].values
            valid_mask = pd.notna(y)
            
            if valid_mask.sum() < 30:
                self.logger.warning(f"日期 {group[date_col].iloc[0]} 的有效数据量: {valid_mask.sum()}，跳过中性化")
                return pd.Series(np.nan, index=group.index)

            temp_x_values = x_vals.loc[group.index[valid_mask]]
            
            model = sm.OLS(y[valid_mask], temp_x_values.astype(float)).fit()
            residuals = pd.Series(np.nan, index=group.index)
            residuals.loc[valid_mask] = model.resid
            return residuals


        for i, (feature_name, is_count_feature) in enumerate(zip(feature_names, mask_list)):
            self.logger.info(f"处理特征 {i+1}/{len(feature_names)}: {feature_name} (类型: {'次数' if is_count_feature else '值'})")
            
            # 预处理
            preprocessed_col = f"{feature_name}_preprocessed"
            if is_count_feature:
                # 截面排序
                all_data[preprocessed_col] = all_data.groupby(date_col)[feature_name].transform(
                    lambda x: x.rank(pct=True, method='average')
                )
            else:
                # 截面Z-score标准化
                all_data[preprocessed_col] = all_data.groupby(date_col)[feature_name].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-10)
                )

            # 中性化
            neutralized_col = f"{feature_name}_neutral"
            
            # 使用 groupby().apply() 替代按日期的for循环
            residuals = all_data.groupby(date_col, group_keys=False).apply(
                _neutralize_single_date, 
                feature_col=preprocessed_col, 
                x_vals=x_values
            )
            all_data[neutralized_col] = residuals
            
            # 删除中间列和原始因子列
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
        
        # 计算指数的未来收益率
        # 先按日期排序，删除重复项，确保每个日期只有一个sh_close
        index_df = all_data[[date_col, 'sh_close']].drop_duplicates(subset=[date_col]).sort_values(by=date_col)
        index_df['index_future_close'] = index_df['sh_close'].shift(-forward_period)
        index_df['index_future_return'] = (index_df['index_future_close'] - index_df['sh_close']) / index_df['sh_close']
        
        # 将指数收益率合并回主数据框
        all_data = pd.merge(all_data, index_df[[date_col, 'index_future_return']], on=date_col, how='left')
        
        # 计算超额收益
        all_data['future_return'] = all_data['future_return'] - all_data['index_future_return']

        # 丢弃多余列
        all_data = all_data.drop(columns=['future_price', 'index_future_close', 'index_future_return'])
        
        # 去除所有pctChg绝对值大于9.85的行
        all_data = all_data[all_data['pctChg'].abs() <= 9.85]
        
        for feature_name in feature_names:
            self.logger.info(f"评估因子: {feature_name}")

            def calculate_ic(group):
                valid_group = group.dropna(subset=[feature_name, 'future_return'])
                if len(valid_group) < 10:
                    self.logger.info(f"在日期 {group[date_col].iloc[0]}, 因子 '{feature_name}' 的有效数据点少于10个，无法计算IC")
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
        self.logger = setup_logger(f"CorrelationTask_{id(self)}")

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

class IRWeightedFactorTask:
    """
    IR加权合并因子任务：
    1. 读取所有 *_neutral 因子
    2. 对每个交易日做截面Z-score标准化
    3. 按IR权重加权合成综合因子分数
    """
    def __init__(self, table_name='neutralized_factors_meta_data', ir_weight_dict=None, output_col='ir_weighted_score'):
        self.table_name = table_name
        self.ir_weight_dict = ir_weight_dict  # 形如 {'因子名': 权重, ...}，因子名不带_neutral
        self.output_col = output_col
        self.logger = setup_logger(f"IRWeightedFactorTask_{id(self)}")

    @with_db_connection
    def run(self, conn, cursor, save_data_path=None):
        self.logger.info(f"读取表 {self.table_name} 进行IR加权合成因子计算...")
        df = pd.read_sql(f'SELECT * FROM {self.table_name}', conn)
        # 1. 获取所有 *_neutral 因子列
        neutral_cols = [col for col in df.columns if col.endswith('_neutral')]
        if not neutral_cols:
            self.logger.error('未找到任何 *_neutral 因子列！')
            return None
        # 2. 过滤出在IR权重字典中的因子

        use_cols = [k for k in self.ir_weight_dict.keys() if k in neutral_cols]
        weights = [self.ir_weight_dict[k] for k in use_cols]

        self.logger.info(f"参与合成的因子: {use_cols}")
        # 3. 对每个交易日做截面Z-score标准化
        date_col = 'date'
        def zscore(group):
            return (group - group.mean()) / (group.std() + 1e-10)
        zscored = df.groupby(date_col)[use_cols].transform(zscore)
        # 4. 加权合成
        score = np.dot(zscored.values, np.array(weights))
        df[self.output_col] = score
        # 5. 可选：保存到数据库
        if save_data_path:
            df.to_sql(save_data_path, if_exists='replace', index=False, con=conn)
            self.logger.info(f"合成因子结果已保存到表 {save_data_path}")
        return df[[date_col, 'code', self.output_col]]
    
class ResultExportTask:
    """
    从指定表提取最新日期的结果并导出为Excel
    """
    def __init__(self, table_name='ir_weighted_factors_meta_data', output_excel='result_latest.xlsx'):
        self.table_name = table_name
        self.output_excel = output_excel
        self.logger = setup_logger(f"ResultExportTask_{id(self)}")

    @with_db_connection
    def run(self, conn, cursor):
        self.logger.info(f"从表 {self.table_name} 读取最新日期数据...")
        df = pd.read_sql(f'SELECT * FROM {self.table_name}', conn)
        if 'date' not in df.columns:
            self.logger.error('数据表中没有date列！')
            return None
        latest_date = df['date'].max()
        latest_df = df[df['date'] == latest_date]
        latest_df.to_excel(self.output_excel, index=False)
        self.logger.info(f"最新日期({latest_date})数据已保存到 {self.output_excel}")
        return latest_df

if __name__ == "__main__":
    # 配置基本日志记录
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # 日志文件名带时间戳
    log_file = get_logfile_with_time()

    df = get_factor_merge_table(factor_names=factor_names)

    # task = FactorTask(all_data=df, cluster_names=['industry'], feature_names=list(factor_dict.keys()), mask_list=[False]*len(factor_dict), log_file=log_file)
    task = FactorTask(all_data=df, cluster_names=['industry'], feature_names=factor_names, mask_list=[mask_dict.get(factor, False) for factor in factor_names], log_file=log_file)
    results, all_data = task.run(save_data_path='neutralized_factors_meta_data')
    print("因子评估结果:")
    for i in results:
        print(f"{i}: {results[i]}")

    # 新增：运行相关性分析任务
    correlation_task = CorrelationTask(table_name='neutralized_factors_meta_data')
    correlation_results = correlation_task.run()
    if correlation_results is not None:
        print("\n中性化因子相关性分析结果 (Top 10):")
        print(correlation_results.head(10))

    # 新增：运行IR加权合成因子任务
    ir_weight_dict = {i: results[i]['IR'] for i in results}
    ir_weighted_task = IRWeightedFactorTask(table_name='neutralized_factors_meta_data', ir_weight_dict=ir_weight_dict)
    ir_weighted_results = ir_weighted_task.run(save_data_path='ir_weighted_factors_meta_data')

    # 新增：导出最新一天结果到Excel
    export_task = ResultExportTask(table_name='ir_weighted_factors_meta_data', output_excel='result_latest.xlsx')
    export_task.run()