
import pandas as pd
import numpy as np
from functools import reduce
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import logging
from sklearn.isotonic import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from tqdm import tqdm
from factors.factor import factor_names, factor_dict, mask_dict, get_factor_merge_table

class FactorTask:
    def __init__(self, all_data, cluster_names=None, feature_names=None, mask_list=None, date_col='date', forward_period=5, log_file="util_main.INFO"):
        self.all_data = all_data
        self.cluster_names = cluster_names if cluster_names is not None else ['industry']
        self.feature_names = feature_names if feature_names is not None else factor_names
        self.mask_list = mask_list if mask_list is not None else [mask_dict[i] for i in self.feature_names]
        self.date_col = date_col
        self.forward_period = forward_period
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)

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
            logging.info(f"处理特征 {i+1}/{len(feature_names)}: {feature_name} (类型: {'次数' if is_count_feature else '值'})")
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
                logging.info(f"处理日期 {date} 的特征 {feature_name}")
                date_mask = all_data[date_col] == date
                date_data = all_data[date_mask]
                y = date_data[preprocessed_col].values
                valid_mask = pd.notna(y)
                if valid_mask.sum() < 30:
                    logging.warning(f"日期 {date} 的有效数据量: {valid_mask.sum()}，跳过中性化")
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
            logging.info(f"评估因子: {feature_name}")

            def calculate_ic(group):
                valid_group = group.dropna(subset=[feature_name, 'future_return'])
                if len(valid_group) < 10:
                    logging.info(f"在日期 {group[date_col].iloc[0]}, 因子 '{feature_name}' 或收益率是常数，无法计算IC")
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
                logging.info(f"因子{feature_name}, 加权IC均值: {ic_mean:.4f}, 加权IC标准差: {ic_std:.4f}, IR: {ir:.4f}")
            else:
                logging.warning(f'无法计算因子 {feature_name} 的IC值 (数据不足或全为NaN)')
        return results, all_data
    
    def run(self, save_data_path=None):
        logging.info(f"在分类变量 {self.cluster_names+['market_cap']} 下处理中性化特征: {self.feature_names}")
        self.process_and_neutralize_factors()
        results, all_data = self.evaluate_neutralized_factors()
        if save_data_path:
            all_data.to_excel(save_data_path)
        return results, all_data

if __name__ == "__main__":
    # 示例用法
    df = get_factor_merge_table()
    task = FactorTask(all_data=df, cluster_names=['industry'], feature_names=list(factor_dict.keys()), mask_list=[False]*len(factor_dict))
    results, all_data = task.run()
    print(results)

