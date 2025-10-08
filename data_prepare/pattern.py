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

# --- 新增：用于并行计算的独立工作函数 (分块任务版) ---
def _calculate_dtw_for_chunk(args):
    """
    计算一个'主'模式与所有过去模式的DTW距离。
    这是一个'大块'任务，以减少进程开销。
    """
    i, patterns_data, dtw_columns = args
    current_pattern = patterns_data[i]
    scaler = StandardScaler()
    
    avg_distances_to_past = []
    distances_per_column = {col: [] for col in dtw_columns}
    # 用于累加每个过去模式的平均距离
    all_past_avg_distances = []

    # 在这个工作函数内部循环，计算与所有过去模式的距离
    for j in range(i):
        past_pattern = patterns_data[j]
        # 用于存储当前模式与这一个过去模式在5个维度上的距离
        pair_distances = []

        for col in dtw_columns:
            series1 = current_pattern['arrays'][col].reshape(-1, 1)
            series2 = past_pattern['arrays'][col].reshape(-1, 1)

            series1_clean = series1[~np.isnan(series1).any(axis=1)]
            series2_clean = series2[~np.isnan(series2).any(axis=1)]

            if len(series1_clean) != 20 or len(series2_clean) != 20:
                continue
            else:
                # 标准化
                # series1_scaled = scaler.fit_transform(series1_clean.reshape(-1, 1)).flatten()
                # series2_scaled = scaler.fit_transform(series2_clean.reshape(-1, 1)).flatten()
                distance, _ = fastdtw(series1_clean, series2_clean, dist=euclidean)

            pair_distances.append(distance)
            distances_per_column[col].append(distance)
        
        # 计算与这一个过去模式的平均距离，并累加
        if pair_distances:
            all_past_avg_distances.append(np.mean(pair_distances))

    # 返回当前模式的ID和完整的计算结果
    return (current_pattern['id'], {
        'code': current_pattern['code'],
        'date': current_pattern['end_date'],
        'avg_dtw_distances': all_past_avg_distances, # 现在这里是与每个过去模式的平均距离列表
        'dtw_distances': distances_per_column
    })

# --- 新增：用于DTW统计的独立工作函数 ---
def _dtw_worker(args):
    """
    计算一个 test_pattern 与所有 finded_patterns 的DTW距离，并返回中位数。
    """
    test_pattern_data, all_finded_patterns_data, dtw_columns = args
    
    distances = []
    for finded_pattern_data in all_finded_patterns_data:
        pair_distances = []
        for col in dtw_columns:
            series1 = test_pattern_data['arrays'][col].reshape(-1, 1)
            series2 = finded_pattern_data['arrays'][col].reshape(-1, 1)

            distance, _ = fastdtw(series1, series2, dist=euclidean)

            pair_distances.append(distance)
        
        # 计算两个模式在所有维度上的平均距离
        if pair_distances:
            avg_pair_distance = np.mean(pair_distances)
            distances.append(avg_pair_distance)

    # 计算当前test_pattern与所有finded_patterns的DTW距离的中位数
    if not distances:
        return None
    
    median_dtw = np.median(distances)
    
    return {
        'code': test_pattern_data['code'],
        'date': test_pattern_data['end_date'],
        'median_dtw': median_dtw
    }

# --- 新增：用于评估涨跌模式相似度的独立工作函数 ---
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

            # 确保数据是20天，否则距离为无穷大
            if len(series1) != 20 or len(series2) != 20:
                continue
            else:
                distance, _ = fastdtw(series1, series2, dist=euclidean)
            pair_distances.append(distance)
        
        if pair_distances:
            avg_pair_distance = np.mean(pair_distances)
            distances.append(avg_pair_distance)

    if not distances:
        return None
    
    return {
        'code': source_pattern_data['code'],
        'date': source_pattern_data['end_date'],
        'mean_dtw': np.mean(distances),
        'median_dtw': np.median(distances),
        'min_dtw': np.min(distances),
        'max_dtw': np.max(distances)
    }

def analyze_pattern_dtw(result_df: pd.DataFrame) -> dict:
    """
    使用动态时间规整（DTW）分析每个模式与过去模式的形态相似性。
    此版本经过优化，通过预处理、并行计算和任务分块来提高性能。

    Args:
        result_df (pd.DataFrame): 包含所有已识别模式的数据，按 'pattern_index' 区分。

    Returns:
        dict: 一个字典，键是 pattern_id，值是包含以下信息的另一个字典：
              - 'code': 模式对应的股票代码。
              - 'date': 模式的结束日期。
              - 'avg_dtw_distances': 一个列表，包含当前模式与所有过去模式的平均DTW距离。
              - 'dtw_distances': 一个字典，包含各特征（如 'close_diff'）与过去模式的DTW距离列表。
    """
    print("开始进行形态DTW分析（优化版）...")
    
    if result_df.empty:
        print("输入的DataFrame为空，无法进行分析。")
        return {}

    # 定义用于DTW计算的列
    dtw_columns = ['open_diff', 'high_diff', 'low_diff', 'close_diff', 'volume_diff']
    
    # # 填充可能存在的NaN值
    # result_df[dtw_columns] = result_df.groupby('pattern_index')[dtw_columns].transform(lambda x: x.dropna().reset_index(drop=True))

    # --- 优化点 1: 数据预处理 ---
    # 一次性将所有模式数据提取并转换为Numpy数组，存储在列表中
    patterns_data = []
    unique_patterns = np.sort(result_df['pattern_index'].unique())
    
    for pid in tqdm(unique_patterns, desc="Preprocessing patterns"):
        pattern_df = result_df[result_df['pattern_index'] == pid]
        pattern_arrays = {}
        for col in dtw_columns:
            pattern_arrays[col] = pattern_df[col].values
        
        patterns_data.append({
            'id': pid,
            'code': pattern_df['code'].iloc[0],
            'end_date': pattern_df['date'].iloc[-1],
            'arrays': pattern_arrays
        })



    # --- 新增：如果模式过多，进行随机采样 ---
    if len(patterns_data) > 1000:
        print(f"模式数量过多 ({len(patterns_data)})，随机采样500个进行分析。")
        # 为了保持原始的时间顺序，我们先对索引进行采样，然后提取对应的元素
        sampled_indices = sorted(random.sample(range(len(patterns_data)), 500))
        patterns_data = [patterns_data[i] for i in sampled_indices]

    # --- 新增：按日期对所有模式进行排序 ---
    patterns_data.sort(key=lambda p: p['end_date'])

    # --- 优化点 2: 任务创建与分块 ---
    # 1. 为并行任务准备参数
    # 每个任务现在是计算一个模式与它之前所有模式的DTW
    # `indices_to_process` 应该是从1到len-1，因为第0个模式没有历史可比较
    indices_to_process = range(1, len(patterns_data))
        
    tasks = [(i, patterns_data, dtw_columns) for i in indices_to_process]

    # 2. 使用进程池并行执行
    results = []
    # 如果模式数量很少，不值得开启多进程
    if len(patterns_data) < 10:
        print("模式数量较少，使用单进程计算。")
        results = [_calculate_dtw_for_chunk(task) for task in tqdm(tasks, desc="Single-process DTW")]
    else:
        with Pool() as pool:
            results = list(tqdm(pool.imap(_calculate_dtw_for_chunk, tasks), total=len(tasks), desc="Parallel DTW (Chunked)"))

    # 3. 直接将结果转换为字典
    dtw_results_dict = dict(results)
            
    return dtw_results_dict

def save_patterns_to_sql(df: pd.DataFrame, db_path: str):
    """
    将 pattern_df 按照每个模式的结束日期进行分组，
    并将每个日期的数据保存到 SQLite 数据库的不同表中。

    Args:
        df (pd.DataFrame): 包含模式数据的 DataFrame，需要有 'pattern_index' 和 'date' 列。
        db_path (str): 输出的 SQLite 数据库文件路径。
    """
    print(f"正在将结果按日期保存到数据库 {db_path}...")
    
    # 找到每个 pattern_index 的结束日期
    end_dates = df.groupby('pattern_index')['date'].max()
    
    # 将结束日期映射回原始 DataFrame
    df['end_date'] = df['pattern_index'].map(end_dates)
    
    # 按结束日期分组
    grouped = df.groupby('end_date')
    
    # 创建数据库连接
    conn = sqlite3.connect(db_path)
    
    try:
        for date, group in tqdm(grouped, desc="Saving patterns to SQL by date"):
            # 将日期格式化为 YYYY_MM_DD 字符串，作为表名
            table_name = f"pattern_{date}"
            # 将 group 数据写入对应的表
            group.to_sql(table_name, conn, if_exists='replace', index=False)
    finally:
        # 关闭数据库连接
        conn.close()
    
    print(f"结果已成功保存到数据库 {db_path}")

def plot_dtw_results_by_date(results_path='dtw_results.pkl'):
    """
    读取DTW结果，按日期排序，计算每个模式与其历史模式的平均DTW距离的均值，并绘制折线图。
    """
    print(f"正在从 {results_path} 加载结果并绘图...")
        
    # 1. 加载数据
    try:
        with open(results_path, 'rb') as f:
            dtw_results = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 未找到结果文件 {results_path}。请先运行DTW分析。")
        return
        
    if not dtw_results:
        print("结果文件为空，无法绘图。")
        return

    # 2. 按照date排列
    # dtw_results.items() -> [(pattern_id, {'code': ..., 'date': ..., 'avg_dtw_distances': [...]}), ...]
    # 我们根据字典中的 'date' 键来排序
    sorted_results = sorted(dtw_results.items(), key=lambda item: item[1]['date'])

    # 3. 取出每个元素的avg_dtw_distances求均值
    mean_avg_distances = []
    dates = []
    for pattern_id, data in sorted_results[-100:]:
        # 过滤掉没有历史记录的模式（avg_dtw_distances为空列表）
        if data['avg_dtw_distances']:
            mean_val = np.mean(data['avg_dtw_distances'])
            mean_avg_distances.append(mean_val)
            dates.append(pd.to_datetime(data['date']))

    if not mean_avg_distances:
        print("没有可供绘图的数据（可能所有模式都没有历史可比较）。")
        return

    # 4. 画一个折线图
    plt.figure(figsize=(15, 7))
    plt.plot(dates, mean_avg_distances, marker='o', linestyle='-', markersize=4)
        
    plt.title('Mean of Average DTW Distances of Patterns (Sorted by Date)')
    plt.xlabel('Date')
    plt.ylabel('Mean of Average DTW Distances')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
        
    # 保存图表到文件
    chart_path = 'dtw_results_by_date.png'
    plt.savefig(chart_path)
    print(f"图表已保存到 {chart_path}")
        
    # 显示图表
    plt.show()

def plot_random_queen_histograms(results_path='dtw_results.pkl', num_queens=500):
    """
    随机选择N个模式（“皇后”），并为每个模式的avg_dtw_distances绘制直方图。
    每个直方图展示了一个模式与其所有历史模式的DTW平均距离的分布情况。
    """
    print(f"正在从 {results_path} 加载结果并为 {num_queens} 个随机皇后绘制直方图...")

    # 1. 加载数据
    try:
        with open(results_path, 'rb') as f:
            dtw_results = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 未找到结果文件 {results_path}。")
        return
    
    if not dtw_results:
        print("结果文件为空，无法绘图。")
        return

    # 2. 随机选择N个“皇后”
    # 过滤掉那些没有历史可比较的模式（即avg_dtw_distances列表为空的）
    valid_patterns = {k: v for k, v in dtw_results.items() if v.get('avg_dtw_distances')}
    
    if len(valid_patterns) == 0:
        print("没有找到包含有效历史距离数据的模式。")
        return

    if len(valid_patterns) < num_queens:
        print(f"有效模式数量 ({len(valid_patterns)}) 少于请求的数量 ({num_queens})，将使用所有有效模式。")
        num_to_sample = len(valid_patterns)
    else:
        num_to_sample = num_queens

    # 从有效模式中随机采样
    random_queens_items = random.sample(list(valid_patterns.items()), num_to_sample)
    
    # 提取出皇后的数据，并按日期排序，使图例更有序
    queens = sorted([item[1] for item in random_queens_items], key=lambda q: q['date'])

    # 3. 绘制直方图
    plt.figure(figsize=(15, 8))
    
    chunk = 10
    for queen in queens[-chunk:]:
        distances = queen['avg_dtw_distances']
        date_str = queen['date']
        
        # 使用alpha透明度让重叠的直方图更易读
        plt.hist(distances, bins=20, alpha=0.6, label=f'Queen from {date_str}')

    plt.title(f'Histograms of Avg DTW Distances for {chunk} Random Queens')
    plt.xlabel('Average DTW Distance to Past Patterns')
    plt.ylabel('Frequency')
    plt.xlim(0, 4)
    # plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存图表到文件
    chart_path = 'dtw_queen_histograms.png'
    plt.savefig(chart_path)
    print(f"皇后直方图已保存到 {chart_path}")

    # 显示图表
    plt.show()

class PatternMatch:
    def __init__(self, df: pd.DataFrame, window: int = 20):
        self.df = df
        self.window = window
        self.match_cols = ['close', 'open', 'high', 'low', 'volume', 'pctChg']
        self.diff_cols = [i+'_diff' for i in self.match_cols]
        self.df = self.df.sort_values(by=['code', 'date']).reset_index(drop=True)
        self.pct_chg_threshold = 5  # pctChg阈值
        self.look_back_period = 60
        self.future_look_period = 10
        self.look_back_chg_count = 3
        self.feature_return = 9.85
        self.finded_patterns = pd.DataFrame()
        self.test_patterns = pd.DataFrame()

    def add_pct_chg(self):
        df = self.df
        # 按code分组计算'close', 'open', 'high', 'low', 'volume'的日度变化
        for col in self.match_cols:
            # groupby('code')保证了每个股票的第一天diff值为NaN
            df[col+'_diff'] = df.groupby('code')[col].pct_change()
        self.df = df
        return df

    def extract_test_pattern(self) -> pd.DataFrame:
        df = self.df
        window = self.window
       
        # 使用tqdm显示进度
        tqdm.pandas(desc="Processing stocks")

        # 提取每个股票的最近20天数据
        test_patterns = df.groupby('code').tail(window+self.look_back_period)
        
        # 选择最近20天pctChg绝对值均小于等于6%的股票
        test_patterns = test_patterns.groupby('code').filter(
            lambda x: (x['pctChg'].tail(window).abs() <= self.pct_chg_threshold).all() and
                  (x['pctChg'].head(self.look_back_period) >= self.pct_chg_threshold).sum() >= self.look_back_chg_count
        )

        test_patterns = test_patterns.sort_values(by=['code', 'date']).reset_index(drop=True)

        test_patterns = df.groupby('code').tail(self.window)

        # 过滤掉数据不足window天或包含NaN值的股票以及包含无穷大值的
        test_patterns = test_patterns.groupby('code').filter(
            lambda x: x[self.diff_cols].notna().all().all() and np.isfinite(x[self.diff_cols]).all().all()
        )

        test_patterns = test_patterns.sort_values(by=['code', 'date']).reset_index(drop=True)

        self.test_patterns = test_patterns
        return test_patterns
    
    def find_pattern_v1(self) -> pd.DataFrame:
        """
        在一个包含所有股票行情的大表中，提取特定模式的数据。

        模式定义：
        1. 连续20天内，每日的涨跌幅（pctChg）绝对值不超过6%。
        2. 在这20天之后的第21天，涨跌幅大于6%。

        Args:
            df (pd.DataFrame): 包含所有股票数据的DataFrame，
                            需要包含 'code', 'date', 'pctChg' 列，
                            并按 'code', 'date' 排序。

        Returns:
            pd.DataFrame: 一个新的DataFrame，包含所有被提取的模式数据，
                        并带有一个 'pattern_index' 列。
        """
        df = self.df
        window = self.window

        # 使用tqdm显示进度
        tqdm.pandas(desc="Processing stocks")

        # 为了在lambda中同时使用'pctChg'和'close'，我们需要对整个DataFrame进行滚动
        # 这比只滚动一个Series要慢，但对于多列逻辑是必需的
        # 我们创建一个辅助函数来处理每个窗口
        lookback_period = self.look_back_period
        future_look_period = self.future_look_period
        total_window = lookback_period + window + future_look_period

        def find_pattern_in_window(w):
            # 重新定义窗口大小以适应新的逻辑
            # w 现在是一个 Series
            if len(w) != total_window:
                return False

            # 将窗口划分为三部分
            lookback_part = w.iloc[0:lookback_period]
            pattern_part = w.iloc[lookback_period : lookback_period + window]
            future_part = w.iloc[lookback_period + window : total_window]

            # 条件0: 过去60天内，涨幅超过阈值的天数至少为3天
            condition0 = (lookback_part >= self.pct_chg_threshold).sum() >= self.look_back_chg_count
            if not condition0:
                return False
            
            # 条件1: 20天稳定期内，每日涨跌幅绝对值 <= 阈值
            condition1 = (pattern_part.abs() <= self.pct_chg_threshold).all()

            if not condition1:
                return False

            # 条件2: 未来10天内，pctChg累计涨幅 > 9.85
            condition2 = future_part.sum() >= self.feature_return
            if not condition2:
                return False

            return True

        # 对每个股票分组应用滚动窗口
        # 使用 progress_apply 来显示进度
        # 只对 pctChg 列进行滚动操作
        pattern_indices = df.groupby('code')['pctChg'].progress_apply(
            lambda x: x.rolling(window=total_window, min_periods=total_window)
                   .apply(find_pattern_in_window, raw=False)
        ).fillna(0).astype(bool)

        # 获取满足条件的窗口的结束索引
        pattern_end_indices = df.index[pattern_indices]

        # 提取所有符合条件的20天数据段
        pattern_dfs = []
        pattern_id = 0
        # --- 修复：索引逻辑调整，end_idx是窗口的最后一天，start_idx是模式的开始 ---
        for end_idx in tqdm(pattern_end_indices, desc="Extracting patterns"):
            # 模式的20天在 total_window 的中间部分
            pattern_start_idx = end_idx - future_look_period - window + 1
            pattern_end_idx = end_idx - future_look_period + 1
            pattern_chunk = df.iloc[pattern_start_idx:pattern_end_idx].copy()
            
            # 确保提取的长度正确
            if len(pattern_chunk) == window:
                pattern_chunk['pattern_index'] = pattern_id
                pattern_dfs.append(pattern_chunk)
                pattern_id += 1

        # 合并所有模式数据
        if not pattern_dfs:
            print("未找到任何满足条件的模式。")
            self.finded_patterns = pd.DataFrame()
            return self.finded_patterns
            
        result_df = pd.concat(pattern_dfs, ignore_index=True)

        # 过滤掉数据不足window天或包含NaN值的模式以及包含无穷大值的模式
        result_df = result_df.groupby('pattern_index').filter(
            lambda x: len(x) == window and x[self.diff_cols].notna().all().all() and np.isfinite(x[self.diff_cols]).all().all()
        )
        result_df = result_df.sort_values(by=['pattern_index', 'date']).reset_index(drop=True)
        self.finded_patterns = result_df
        print("找到的模式数量:", result_df['pattern_index'].nunique())
        return result_df

    def calculate_dtw_stats(self) -> pd.DataFrame:
        """
        分布式计算每个test_pattern与所有finded_patterns的DTW距离，并统计中位数。
        """
        print("开始计算测试模式与历史模式的DTW统计...")

        dtw_columns = self.diff_cols

        # --- 数据预处理 ---
        # 1. 预处理 test_patterns
        test_patterns_data = []
        for code, group in tqdm(self.test_patterns.groupby('code'), desc="Preprocessing test patterns"):
            pattern_arrays = {col: group[col].values for col in dtw_columns}
            test_patterns_data.append({
                'code': code,
                'end_date': group['date'].iloc[-1],
                'arrays': pattern_arrays
            })

        # 2. 预处理 finded_patterns
        finded_patterns_data = []
        for pid in tqdm(self.finded_patterns['pattern_index'].unique(), desc="Preprocessing finded patterns"):
            pattern_df = self.finded_patterns[self.finded_patterns['pattern_index'] == pid]
            pattern_arrays = {col: pattern_df[col].values for col in dtw_columns}
            finded_patterns_data.append({'arrays': pattern_arrays})

        # --- 创建并行任务 ---
        tasks = [(tp_data, finded_patterns_data, dtw_columns) for tp_data in test_patterns_data]

        # --- 使用进程池并行执行 ---
        results = []
        with Pool() as pool:
            results = list(tqdm(pool.imap(_dtw_worker, tasks), total=len(tasks), desc="Calculating DTW medians"))

        # --- 整理结果 ---
        valid_results = [res for res in results if res is not None]
        if not valid_results:
            print("没有计算出有效的DTW结果。")
            return pd.DataFrame()

        dtw_stats_df = pd.DataFrame(valid_results)
        dtw_stats_df = dtw_stats_df.sort_values(by='median_dtw').reset_index(drop=True)
        
        # 打印DTW中位数距离的统计信息
        mean_dtw = dtw_stats_df['median_dtw'].mean()
        std_dtw = dtw_stats_df['median_dtw'].std()
        print(f"DTW中位数距离的均值: {mean_dtw:.4f}")
        print(f"DTW中位数距离的标准差: {std_dtw:.4f}")
            
        print("DTW统计计算完成。")
        return dtw_stats_df

    def find_latent_patterns(self, rise_threshold=1.0985, fall_threshold=0.95, future_window=10):
        """
        重构潜伏模式提取逻辑。
        1. 识别当前pctChg绝对值大于5的日期（T_end）。
        2. 往前找到上一个pctChg绝对值大于5的日期（T_start）。
        3. 中间序列为潜伏期，需至少20天。
        4. 如果超过20天，只取最近20天。
        5. 根据潜伏期结束后的未来10天表现，标记为“上涨”或“下跌”模式。
        """
        print("正在使用新逻辑提取潜伏模式...")
        self.add_pct_chg() # 确保pctChg列存在
        
        all_rise_patterns = []
        all_fall_patterns = []
        pattern_id = 0

        # 对每支股票进行独立分析
        for code, group in tqdm(self.df.groupby('code'), desc="Finding latent patterns"):
            # 找到所有波动剧烈的点的索引
            volatility_indices = group.index[group['pctChg'].abs() > self.pct_chg_threshold]

            if len(volatility_indices) < 2:
                continue

            # 遍历每两个连续的剧烈波动点
            for i in range(len(volatility_indices) - 1):
                t_start_idx = volatility_indices[i]
                t_end_idx = volatility_indices[i+1]

                # 潜伏期是两个波动点之间的序列
                latent_period_full = group.loc[t_start_idx + 1 : t_end_idx - 1]
                
                # 条件1: 潜伏期至少20天
                if len(latent_period_full) >= self.window:
                    # 条件2: 如果超长，只取最近20天
                    pattern_df = latent_period_full.tail(self.window).copy()

                    # 检查提取的20天pattern是否有效
                    if len(pattern_df) != self.window or pattern_df[self.diff_cols].isnull().values.any() or pattern_df[self.diff_cols].isin([np.inf, -np.inf]).values.any():
                        continue

                    # --- 检查未来表现 ---
                    # 潜伏期结束点是 t_end_idx - 1
                    future_start_loc = group.index.get_loc(t_end_idx)
                    
                    # 检查是否有足够的未来数据
                    if future_start_loc + future_window > len(group):
                        continue
                    
                    future_df = group.iloc[future_start_loc : future_start_loc + future_window]
                    pct_chg_ratio = 1 + future_df['pctChg'] / 100
                    cumulative_product = pct_chg_ratio.prod()

                    pattern_df['pattern_index'] = pattern_id
                    
                    # 判断是上涨还是下跌模式
                    if cumulative_product >= rise_threshold:
                        all_rise_patterns.append(pattern_df)
                        pattern_id += 1
                    elif cumulative_product <= fall_threshold:
                        all_fall_patterns.append(pattern_df)
                        pattern_id += 1

        rise_df = pd.concat(all_rise_patterns, ignore_index=True) if all_rise_patterns else pd.DataFrame()
        fall_df = pd.concat(all_fall_patterns, ignore_index=True) if all_fall_patterns else pd.DataFrame()

        print(f"找到 {rise_df['pattern_index'].nunique()} 个上涨模式和 {fall_df['pattern_index'].nunique()} 个下跌模式。")
        
        return rise_df, fall_df

    def evaluate_rise_fall_similarity(self):
        """
        使用fastdtw分布式计算，评估所有“潜伏后上升”模式和“潜伏后下跌”模式之间的相似度与差异性。
        (已更新为使用新的 find_latent_patterns 方法)
        """
        print("开始评估 '潜伏后上升' vs '潜伏后下跌' 模式的相似性...")

        # 1. 使用新方法直接获取上涨和下跌模式
        rise_patterns_df, fall_patterns_df = self.find_latent_patterns()

        if rise_patterns_df.empty or fall_patterns_df.empty:
            print("错误：上涨模式或下跌模式不足，无法进行比较。")
            return

        # 2. 提取模式数据
        dtw_columns = self.diff_cols

        def extract_patterns_data(df, pattern_type):
            patterns_data = []
            for pid in tqdm(df['pattern_index'].unique(), desc=f"Preprocessing {pattern_type} patterns"):
                pattern_df = df[df['pattern_index'] == pid]
                pattern_arrays = {col: pattern_df[col].values for col in dtw_columns}
                patterns_data.append({
                    'code': pattern_df['code'].iloc[-1],
                    'end_date': pattern_df['date'].iloc[-1],
                    'arrays': pattern_arrays
                })
            return patterns_data

        rise_patterns_data = extract_patterns_data(rise_patterns_df, "rise")
        fall_patterns_data = extract_patterns_data(fall_patterns_df, "fall")

        if not rise_patterns_data or not fall_patterns_data:
            print("错误：有效数据提取后，上升或下跌模式为空。")
            return

        # 3. 创建并行任务 (与之前相同)
        tasks1 = [(rp_data, fall_patterns_data, dtw_columns) for rp_data in rise_patterns_data]
        tasks2 = [(fp_data, rise_patterns_data, dtw_columns) for fp_data in fall_patterns_data]
        tasks3 = [(rise_patterns_data[i], rise_patterns_data[:i] + rise_patterns_data[i+1:], dtw_columns) for i in range(len(rise_patterns_data))]
        tasks4 = [(fall_patterns_data[i], fall_patterns_data[:i] + fall_patterns_data[i+1:], dtw_columns) for i in range(len(fall_patterns_data))]

        # 4. 执行并行计算 (与之前相同)
        with Pool() as pool:
            print("\n计算 '上升模式' vs '所有下跌模式' 的DTW...")
            results1 = list(tqdm(pool.imap(_similarity_worker, tasks1), total=len(tasks1)))
            
            print("计算 '下跌模式' vs '所有上升模式' 的DTW...")
            results2 = list(tqdm(pool.imap(_similarity_worker, tasks2), total=len(tasks2)))

            print("计算 '上升模式' vs '其他上升模式' 的DTW (内部相似度)...")
            results3 = list(tqdm(pool.imap(_similarity_worker, tasks3), total=len(tasks3)))

            print("计算 '下跌模式' vs '其他下跌模式' 的DTW (内部相似度)...")
            results4 = list(tqdm(pool.imap(_similarity_worker, tasks4), total=len(tasks4)))

        # 5. 汇总和分析结果 (与之前相同)
        df1 = pd.DataFrame([r for r in results1 if r])
        df2 = pd.DataFrame([r for r in results2 if r])
        df3 = pd.DataFrame([r for r in results3 if r])
        df4 = pd.DataFrame([r for r in results4 if r])

        print("\n--- 相似性评估结果 ---")
        if not df3.empty:
            print(f"上升模式内部平均DTW距离 (相似度): {df3['mean_dtw'].mean():.4f}")
        if not df4.empty:
            print(f"下跌模式内部平均DTW距离 (相似度): {df4['mean_dtw'].mean():.4f}")
        if not df1.empty:
            print(f"上升模式与下跌模式的平均DTW距离 (差异度): {df1['mean_dtw'].mean():.4f}")
        if not df2.empty:
            print(f"下跌模式与上升模式的平均DTW距离 (差异度): {df2['mean_dtw'].mean():.4f}")
        
        print("\n结论:")
        if not df1.empty and not df3.empty and not df1.empty and not df4.empty:
            rise_intra_dist = df3['mean_dtw'].mean()
            rise_inter_dist = df1['mean_dtw'].mean()
            fall_intra_dist = df4['mean_dtw'].mean()
            fall_inter_dist = df2['mean_dtw'].mean()

            if rise_inter_dist > rise_intra_dist:
                print(f" - 上升模式与自身的相似度({rise_intra_dist:.4f}) 高于 与下跌模式的相似度({rise_inter_dist:.4f})。模式有显著差异。")
            else:
                print(f" - 上升模式与自身的相似度({rise_intra_dist:.4f}) 低于或等于 与下跌模式的相似度({rise_inter_dist:.4f})。模式形态相似。")
            
            if fall_inter_dist > fall_intra_dist:
                print(f" - 下跌模式与自身的相似度({fall_intra_dist:.4f}) 高于 与上涨模式的相似度({fall_inter_dist:.4f})。模式有显著差异。")
            else:
                print(f" - 下跌模式与自身的相似度({fall_intra_dist:.4f}) 低于或等于 与上涨模式的相似度({fall_inter_dist:.4f})。模式形态相似。")

        return {"rise_vs_fall": df1, "fall_vs_rise": df2, "rise_vs_rise": df3, "fall_vs_fall": df4}
    
    def main(self):
        self.add_pct_chg()
        self.extract_test_pattern()
        self.find_pattern_v1()
        dtw_stats = self.calculate_dtw_stats()
        print("DTW统计结果：")
        print(dtw_stats.head())
        # pickle保存结果
        with open('dtw_results.pkl', 'wb') as f:
            pickle.dump(dtw_stats.to_dict(orient='records'), f)
        return dtw_stats

if __name__ == "__main__":
    # 示例用法
    from prepare import get_stock_merge_table
    df = get_stock_merge_table(220)
    pattern_matcher = PatternMatch(df, window=30)
    # dtw_stats = pattern_matcher.main() # 注释掉原来的main调用

    # --- 新增：调用相似性评估方法 ---
    similarity_results = pattern_matcher.evaluate_rise_fall_similarity()

    # 你可以在这里添加代码来处理或保存 similarity_results
    if similarity_results:
        print("\n--- 保存相似性评估结果 ---")
        for name, result_df in similarity_results.items():
            if not result_df.empty:
                file_path = f"{name}_similarity_results.csv"
                result_df.to_csv(file_path, index=False)
                print(f"结果已保存到 {file_path}")

    # 从文件加载DTW统计结果并进行分析
    # print("\n--- 从文件加载并分析DTW统计结果 ---")
    # try:
    #     with open('dtw_results.pkl', 'rb') as f:
    #         # 加载数据，它是一个字典列表
    #         dtw_stats_list = pickle.load(f)
        
    #     # 将字典列表转换为DataFrame
    #     dtw_stats_df = pd.DataFrame(dtw_stats_list)
        
    #     if not dtw_stats_df.empty:
    #         print("成功加载DTW统计结果。")
            
    #         # 计算并打印均值和标准差
    #         mean_dtw = dtw_stats_df['median_dtw'].mean()
    #         std_dtw = dtw_stats_df['median_dtw'].std()
            
    #         print(f"DTW中位数距离的均值: {mean_dtw:.4f}")
    #         print(f"DTW中位数距离的标准差: {std_dtw:.4f}")
            
    #         print("\nDTW统计结果 (按距离从小到大排序):")
    #         print(dtw_stats_df.head())
    #     else:
    #         print("加载的DTW结果文件为空或格式不正确。")

    # except FileNotFoundError:
    #     print("错误: 未找到 'dtw_results.pkl' 文件。请确保之前的步骤已成功运行并生成了该文件。")
    # except Exception as e:
    #     print(f"加载或处理文件时发生错误: {e}")
