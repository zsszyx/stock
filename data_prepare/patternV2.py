import pandas as pd
from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import json
import sqlite3
import pickle
from multiprocessing import Pool
import random

def _similarity_worker(args):
    """
    计算一个源模式与一组目标模式之间的DTW距离统计。
    """
    source_pattern_data, target_patterns_data, dtw_columns = args
    
    distances = []
    for target_pattern_data in target_patterns_data:
        pair_distances = []
        for col in dtw_columns:
            series1 = source_pattern_data['arrays'][col].reshape(-1, 1)
            series2 = target_pattern_data['arrays'][col].reshape(-1, 1)

            distance, _ = fastdtw(series1, series2, dist=euclidean)
            pair_distances.append(distance)
        
        if pair_distances:
            avg_pair_distance = np.mean(pair_distances)
            distances.append(avg_pair_distance)

    if not distances:
        return None
    
    return {
        'code': source_pattern_data['code'],
        'date': source_pattern_data['date'],
        'mean_dtw': np.mean(distances),
        'median_dtw': np.median(distances),
        'min_dtw': np.min(distances),
        'max_dtw': np.max(distances)
    }

def _calculate_dtw_for_sets(source_patterns_data, target_patterns_data, dtw_columns):
    """辅助函数，计算两组模式之间的平均DTW距离。"""
    if not source_patterns_data or not target_patterns_data:
        return None
    
    total_distances = []
    for source_pattern in source_patterns_data:
        distances_to_all_targets = []
        for target_pattern in target_patterns_data:
            # 如果源和目标是同一个模式，则跳过（用于内部比较）
            if source_pattern.get('id') is not None and source_pattern.get('id') == target_pattern.get('id'):
                continue

            pair_distances = []
            for col in dtw_columns:
                series1 = source_pattern['arrays'][col].reshape(-1, 1)
                series2 = target_pattern['arrays'][col].reshape(-1, 1)
                distance, _ = fastdtw(series1, series2, dist=euclidean)
                pair_distances.append(distance)
            
            if pair_distances:
                avg_pair_distance = np.mean(pair_distances)
                distances_to_all_targets.append(avg_pair_distance)
        
        if distances_to_all_targets:
            # 计算一个源模式与所有目标模式的平均距离
            total_distances.append(np.mean(distances_to_all_targets))

    return np.mean(total_distances) if total_distances else None

def _evaluate_stock_similarity_worker(args):
    """
    新的工作函数：针对单只股票，评估其内部涨跌模式的相似性。
    """
    code, rise_patterns_df, fall_patterns_df, dtw_columns = args

    def extract_data(df):
        """从DataFrame中提取用于DTW计算的数据结构。"""
        patterns_data = []
        for pid in df['pattern_id'].unique():
            pattern_df = df[df['pattern_id'] == pid]
            patterns_data.append({
                'id': pid, # 添加唯一ID用于内部比较时跳过自己
                'arrays': {col: pattern_df[col].values for col in dtw_columns}
            })
        return patterns_data

    rise_data = extract_data(rise_patterns_df)
    fall_data = extract_data(fall_patterns_df)

    # 1. 计算上涨模式与下跌模式的差异度
    rise_vs_fall_dist = _calculate_dtw_for_sets(rise_data, fall_data, dtw_columns)
    
    # 2. 计算上涨模式的内部相似度
    rise_intra_dist = _calculate_dtw_for_sets(rise_data, rise_data, dtw_columns)

    # 3. 计算下跌模式的内部相似度
    fall_intra_dist = _calculate_dtw_for_sets(fall_data, fall_data, dtw_columns)

    return {
        'code': code,
        'rise_patterns_count': len(rise_data),
        'fall_patterns_count': len(fall_data),
        'rise_vs_fall_dist': rise_vs_fall_dist,
        'rise_intra_dist': rise_intra_dist,
        'fall_intra_dist': fall_intra_dist
    }

class PatternMatch:
    def __init__(self, df: pd.DataFrame, window: int = 20):
        self.df = df
        self.window = window
        self.match_cols = ['close', 'open', 'high', 'low', 'volume']
        # self.diff_cols = [i+'_diff' for i in self.match_cols]
        self.diff_cols = ['close_bb', 'volume_bb']  # 只用部分列进行DTW计算
        self.df = self.df.sort_values(by=['code', 'date']).reset_index(drop=True)

    def add_pct_chg(self):
        df = self.df
        # 按code分组计算'close', 'open', 'high', 'low', 'volume'的日度变化
        for col in self.match_cols:
            # groupby('code')保证了每个股票的第一天diff值为NaN
            df[col+'_diff'] = df.groupby('code')[col].pct_change()
        self.df = df
        return df
    
    def add_zscore(self, window=60):
        """
        按股票代码分组，计算指定窗口大小的滚动Z-score。
        """
        df = self.df
        print(f"正在按 {window} 天窗口计算Z-score...")
        
        # 定义一个函数来计算滚动Z-score
        def rolling_zscore(series):
            rolling_mean = series.rolling(window=window, min_periods=1).mean()
            rolling_std = series.rolling(window=window, min_periods=1).std()
            # 加上一个极小值以避免除以零
            return (series - rolling_mean) / (rolling_std + 1e-8)

        # 对所有match_cols列应用滚动Z-score计算
        for col in tqdm(self.match_cols, desc="计算Z-score"):
            # 使用groupby和transform确保按每只股票独立计算
            df[col+'_diff'] = df.groupby('code')[col].transform(rolling_zscore)

        self.df = df
        return df

    def add_bolling(self, period=20):
        df = self.df
        print("正在计算布林带指标...")
        
        # 计算布林带指标
        def compute_bollinger_bands(group, col='close'):
            rolling_mean = group[col].rolling(window=period, min_periods=20).mean()
            rolling_std = group[col].rolling(window=period, min_periods=20).std()
            group[f'{col}_bb'] = rolling_std/(rolling_mean+1e-8)
            return group

        df = df.groupby('code').apply(compute_bollinger_bands, col='close').reset_index(drop=True)
        df = df.groupby('code').apply(compute_bollinger_bands, col='volume').reset_index(drop=True)
        self.df = df
        return df
    
    def extract_event_driven_patterns(self, threshold: float = 5.0, min_len: int = 10, max_len: int = 30):
        """
        对每个表格按照code group之后提取两个pctChg绝对值大于阈值之间的序列，
        取最后10-30个尽可能长的非空无inf序列，并根据最后pctchg是涨还是跌区分出上涨和下跌模式。

        Args:
            threshold (float): pctChg绝对值的波动阈值。
            min_len (int): 模式的最小长度。
            max_len (int): 模式的最大长度。

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 包含所有上涨模式和下跌模式的两个DataFrame。
        """
        print("开始使用事件驱动逻辑提取模式...")
        
        all_rise_patterns = []
        all_fall_patterns = []
        pattern_id = 0

        # 对每支股票进行独立分析
        for code, group in tqdm(self.df.groupby('code'), desc="分析股票"):
            # 找到所有“事件”点的索引（pctChg剧烈波动）
            event_indices = group.index[group['pctChg'].abs() > threshold]

            if len(event_indices) < 2:
                continue

            # 遍历每两个连续的事件点
            for i in range(len(event_indices) - 1):
                start_event_idx = event_indices[i]
                end_event_idx = event_indices[i+1]

                # 提取两个事件点之间的“平静期”序列
                latent_period_full = group.loc[start_event_idx + 1 : end_event_idx - 1]
                
                # 如果平静期长度小于最小模式长度，则跳过
                if len(latent_period_full) < min_len:
                    continue

                # 从平静期末尾开始，尝试找到最长的有效模式（在[min_len, max_len]范围内）
                for length in range(min(len(latent_period_full), max_len), min_len - 1, -1):
                    pattern_candidate = latent_period_full.tail(length)
                    
                    # 检查数据是否干净（无NaN, 无inf）
                    is_clean = not (pattern_candidate[self.diff_cols].isnull().values.any() or \
                                    np.isinf(pattern_candidate[self.diff_cols]).values.any())

                    if is_clean:
                        # 找到了最长的有效模式
                        final_pattern = pattern_candidate.copy()
                        final_pattern['pattern_id'] = pattern_id
                        final_pattern['pattern_len'] = length
                        
                        # 定义未来N天的窗口来判断涨跌趋势
                        future_window = 5
                        future_start_index = end_event_idx
                        future_end_index_loc = group.index.get_loc(future_start_index) + future_window
                        
                        # 确保未来窗口不超出group的边界
                        if future_end_index_loc <= len(group.index):
                            future_slice = group.iloc[group.index.get_loc(future_start_index):future_end_index_loc]
                            future_pct_chg_sum = future_slice['pctChg'].sum()

                            if future_pct_chg_sum > 5:
                                all_rise_patterns.append(final_pattern)
                            elif future_pct_chg_sum < -5:
                                all_fall_patterns.append(final_pattern)
                        
                        pattern_id += 1
                        break # 找到后就不用再尝试更短的长度了
                
        # 合并所有找到的模式
        rise_df = pd.concat(all_rise_patterns, ignore_index=True) if all_rise_patterns else pd.DataFrame()
        fall_df = pd.concat(all_fall_patterns, ignore_index=True) if all_fall_patterns else pd.DataFrame()

        print(f"提取完成：找到 {rise_df['pattern_id'].nunique() if not rise_df.empty else 0} 个上涨模式和 {fall_df['pattern_id'].nunique() if not fall_df.empty else 0} 个下跌模式。")
        
        self.rise_df = rise_df
        self.fall_df = fall_df
        return rise_df, fall_df

    def evaluate_rise_fall_similarity_code(self, min_patterns_per_stock=2):
        """
        重构版：按 'code' 维度，并行评估每只股票内部的涨跌模式相似性。
        """
        print("开始按股票代码评估模式相似性...")

        if not hasattr(self, 'rise_df') or not hasattr(self, 'fall_df'):
            self.extract_event_driven_patterns()

        rise_df, fall_df = self.rise_df, self.fall_df

        if rise_df.empty or fall_df.empty:
            print("错误：上涨或下跌模式不足，无法进行比较。")
            return pd.DataFrame()

        # 1. 找到同时具有上涨和下跌模式的股票代码
        rise_codes = set(rise_df['code'].unique())
        fall_codes = set(fall_df['code'].unique())
        common_codes = list(rise_codes.intersection(fall_codes))
        
        print(f"找到 {len(common_codes)} 只同时包含涨跌模式的股票。")

        # 2. 创建任务列表，每个任务是一只股票
        tasks = []
        for code in common_codes:
            rise_patterns_for_code = rise_df[rise_df['code'] == code]
            fall_patterns_for_code = fall_df[fall_df['code'] == code]
            
            # 确保每种模式都有足够的数量进行有意义的比较
            if len(rise_patterns_for_code['pattern_id'].unique()) >= min_patterns_per_stock and \
               len(fall_patterns_for_code['pattern_id'].unique()) >= min_patterns_per_stock:
                tasks.append((code, rise_patterns_for_code, fall_patterns_for_code, self.diff_cols))

        if not tasks:
            print(f"没有股票同时拥有至少 {min_patterns_per_stock} 个上涨和下跌模式，无法继续分析。")
            return pd.DataFrame()
            
        print(f"将对 {len(tasks)} 只符合条件的股票进行并行分析...")

        # 3. 使用进程池并行执行
        results = []
        with Pool() as pool:
            results = list(tqdm(pool.imap(_evaluate_stock_similarity_worker, tasks), total=len(tasks), desc="评估个股模式"))

        # 4. 汇总结果并进行分析
        results_df = pd.DataFrame([res for res in results if res is not None])
        if results_df.empty:
            print("没有计算出有效的评估结果。")
            return pd.DataFrame()

        # 计算一个区分度指标
        # 指标 > 1 表示模式间差异大于内部差异，可区分
        # 指标 < 1 表示模式间差异小于内部差异，难区分
        results_df['rise_distinction'] = results_df['rise_vs_fall_dist'] / results_df['rise_intra_dist']
        results_df['fall_distinction'] = results_df['rise_vs_fall_dist'] / results_df['fall_intra_dist']
        
        print("\n--- 个股模式评估结果摘要 ---")
        print(results_df.head())

        # 打印一些统计结论
        distinguishable_rise = results_df[results_df['rise_distinction'] > 1.1]
        distinguishable_fall = results_df[results_df['fall_distinction'] > 1.1]
        
        print(f"\n在 {len(results_df)} 只股票中:")
        print(f" - {len(distinguishable_rise)} 只股票的上涨模式具有显著可区分性 (差异度 > 内部相似度10%以上)。")
        print(f" - {len(distinguishable_fall)} 只股票的下跌模式具有显著可区分性 (差异度 > 内部相似度10%以上)。")

        return results_df.sort_values(by='rise_distinction', ascending=False)

    def evaluate_global_similarity(self, sample_size=200):
        """
        分析所有上涨模式和所有下跌模式的整体差异性。
        """
        print("\n" + "="*30)
        print("开始全局评估所有涨跌模式的整体差异性...")
        print("="*30)

        if not hasattr(self, 'rise_df') or not hasattr(self, 'fall_df'):
            print("错误：尚未提取模式，请先运行 extract_event_driven_patterns。")
            return

        rise_df, fall_df = self.rise_df.copy(), self.fall_df.copy()

        # 如果模式总数过多，进行随机抽样以加快计算速度
        if len(rise_df['pattern_id'].unique()) > sample_size:
            print(f"上涨模式总数超过 {sample_size}，进行随机抽样...")
            sampled_ids = random.sample(list(rise_df['pattern_id'].unique()), sample_size)
            rise_df = rise_df[rise_df['pattern_id'].isin(sampled_ids)]
        
        if len(fall_df['pattern_id'].unique()) > sample_size:
            print(f"下跌模式总数超过 {sample_size}，进行随机抽样...")
            sampled_ids = random.sample(list(fall_df['pattern_id'].unique()), sample_size)
            fall_df = fall_df[fall_df['pattern_id'].isin(sampled_ids)]

        if rise_df.empty or fall_df.empty:
            print("错误：抽样后上涨或下跌模式为空，无法进行全局比较。")
            return

        print(f"将使用 {rise_df['pattern_id'].nunique()} 个上涨模式和 {fall_df['pattern_id'].nunique()} 个下跌模式进行全局评估。")

        # 提取用于DTW计算的数据结构
        def extract_data(df, p_type):
            patterns_data = []
            for pid in tqdm(df['pattern_id'].unique(), desc=f"Preprocessing global {p_type} patterns"):
                pattern_df = df[df['pattern_id'] == pid]
                patterns_data.append({
                    'code': pattern_df['code'].iloc[-1],
                    'date': pattern_df['date'].iloc[-1],
                    'arrays': {col: pattern_df[col].values for col in self.diff_cols}
                })
            return patterns_data

        rise_patterns_data = extract_data(rise_df, "rise")
        fall_patterns_data = extract_data(fall_df, "fall")

        # 创建并行任务
        tasks_rise_vs_fall = [(rp_data, fall_patterns_data, self.diff_cols) for rp_data in rise_patterns_data]
        tasks_rise_intra = [(rise_patterns_data[i], rise_patterns_data[:i] + rise_patterns_data[i+1:], self.diff_cols) for i in range(len(rise_patterns_data))]
        tasks_fall_intra = [(fall_patterns_data[i], fall_patterns_data[:i] + fall_patterns_data[i+1:], self.diff_cols) for i in range(len(fall_patterns_data))]

        # 执行并行计算
        with Pool() as pool:
            print("\n计算全局 '上涨模式' vs '所有下跌模式' 的DTW...")
            results_rise_vs_fall = list(tqdm(pool.imap(_similarity_worker, tasks_rise_vs_fall), total=len(tasks_rise_vs_fall)))
            
            print("计算全局 '上涨模式' 内部DTW...")
            results_rise_intra = list(tqdm(pool.imap(_similarity_worker, tasks_rise_intra), total=len(tasks_rise_intra)))

            print("计算全局 '下跌模式' 内部DTW...")
            results_fall_intra = list(tqdm(pool.imap(_similarity_worker, tasks_fall_intra), total=len(tasks_fall_intra)))

        # 汇总和分析结果
        df_rise_vs_fall = pd.DataFrame([r for r in results_rise_vs_fall if r])
        df_rise_intra = pd.DataFrame([r for r in results_rise_intra if r])
        df_fall_intra = pd.DataFrame([r for r in results_fall_intra if r])

        if df_rise_vs_fall.empty or df_rise_intra.empty or df_fall_intra.empty:
            print("错误：全局评估计算未能生成有效结果。")
            return

        # 计算最终的平均距离
        mean_rise_vs_fall_dist = df_rise_vs_fall['mean_dtw'].mean()
        mean_rise_intra_dist = df_rise_intra['mean_dtw'].mean()
        mean_fall_intra_dist = df_fall_intra['mean_dtw'].mean()

        print("\n--- 全局模式评估结果 ---")
        print(f"上涨模式内部平均距离 (相似度): {mean_rise_intra_dist:.4f}")
        print(f"下跌模式内部平均距离 (相似度): {mean_fall_intra_dist:.4f}")
        print(f"上涨模式与下跌模式平均距离 (差异度): {mean_rise_vs_fall_dist:.4f}")

        print("\n全局结论:")
        if mean_rise_vs_fall_dist > mean_rise_intra_dist:
            print(f" - 全局来看，上涨模式与下跌模式的形态差异 ({mean_rise_vs_fall_dist:.4f}) 大于其内部形态差异 ({mean_rise_intra_dist:.4f})，表明存在可区分的宏观范式。")
        else:
            print(" - 全局来看，上涨模式与下跌模式的形态差异不显著，难以从宏观上区分。")

        return {
            "rise_vs_fall_dist": mean_rise_vs_fall_dist,
            "rise_intra_dist": mean_rise_intra_dist,
            "fall_intra_dist": mean_fall_intra_dist
        }
    
if __name__ == "__main__":
    # 示例用法
    from prepare import get_stock_merge_table
    df = get_stock_merge_table(220)
    pattern_matcher = PatternMatch(df, window=30)
    pattern_matcher.add_bolling()
    pattern_matcher.extract_event_driven_patterns()
    pattern_matcher.evaluate_global_similarity()
