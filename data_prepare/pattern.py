import pandas as pd
from tqdm import tqdm
from dtw import dtw
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import json
import sqlite3

def find_pattern_v1(df: pd.DataFrame, window=20) -> pd.DataFrame:
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
    # 确认数据已排序
    df = df.sort_values(by=['code', 'date']).reset_index(drop=True)

    # 按code分组计算'close', 'open', 'high', 'low', 'volume'的日度变化
    diff_cols = ['close', 'open', 'high', 'low', 'volume']
    for col in diff_cols:
        # groupby('code')保证了每个股票的第一天diff值为NaN
        df[f'{col}_diff'] = df.groupby('code')[col].pct_change()

    # 使用tqdm显示进度
    tqdm.pandas(desc="Processing stocks")

    # 使用groupby和rolling来识别模式
    # 窗口大小为21，其中前20天用于条件1，第21天用于条件2
    pattern_indices = df.groupby('code')['pctChg'].progress_apply(
        lambda x: x.rolling(window=window+1, closed='right', min_periods=window+1).apply(
            lambda w: (w.iloc[-1] > 6) and (abs(w.iloc[:-1]) <= 6).all(),
            raw=False
        )
    ).fillna(0).astype(bool)

    # 获取满足条件的21天窗口的起始索引
    pattern_start_indices = df.index[pattern_indices]

    # 提取所有符合条件的20天数据段
    pattern_dfs = []
    pattern_id = 0
    for end_idx in tqdm(pattern_start_indices, desc="Extracting patterns"):
        start_idx = end_idx - window
        pattern_chunk = df.iloc[start_idx:end_idx].copy()
        pattern_chunk['pattern_index'] = pattern_id
        pattern_dfs.append(pattern_chunk)
        pattern_id += 1

    # 合并所有模式数据
    result_df = pd.concat(pattern_dfs, ignore_index=True)

    return result_df

def analyze_pattern_dtw(result_df: pd.DataFrame) -> dict:
    """
    使用动态时间规整（DTW）分析每个模式与过去模式的形态相似性。

    Args:
        result_df (pd.DataFrame): 包含所有已识别模式的数据，按 'pattern_index' 区分。

    Returns:
        dict: 一个字典，键是 pattern_id，值是包含以下信息的另一个字典：
              - 'code': 模式对应的股票代码。
              - 'date': 模式的结束日期。
              - 'avg_dtw_distances': 一个列表，包含当前模式与所有过去模式的平均DTW距离。
              - 'dtw_distances': 一个字典，包含各特征（如 'close_diff'）与过去模式的DTW距离列表。
    """
    print("开始进行形态DTW分析...")
    
    # 确保数据按 pattern_index 和 date 排序
    result_df = result_df.sort_values(by=['pattern_index', 'date']).reset_index(drop=True)
    
    # 定义用于DTW计算的列
    dtw_columns = ['open_diff', 'high_diff', 'low_diff', 'close_diff', 'volume_diff']
    
    # 填充可能存在的NaN值
    result_df[dtw_columns] = result_df[dtw_columns].fillna(0)

    # 最终存储结果的字典
    dtw_results_dict = {}

    # 获取所有唯一的 pattern_index
    unique_patterns = result_df['pattern_index'].unique()

    # 使用tqdm显示DTW分析的进度
    for current_pattern_id in tqdm(unique_patterns, desc="Analyzing DTW for patterns"):
        # 提取当前模式的数据
        current_pattern_df = result_df[result_df['pattern_index'] == current_pattern_id]
        current_start_date = current_pattern_df['date'].iloc[0]
        
        # 获取当前模式的股票代码和结束日期
        current_code = current_pattern_df['code'].iloc[0]
        current_end_date = current_pattern_df['date'].iloc[-1]

        # 初始化存储与过去模式比较结果的列表
        avg_distances_to_past = []
        # 初始化存储各维度距离的字典
        distances_per_column = {col: [] for col in dtw_columns}

        # 识别过去的模式（结束日期在当前模式开始日期之前）
        past_patterns_df = result_df[result_df['date'] < current_start_date]
        past_pattern_ids = past_patterns_df['pattern_index'].unique()

        # 遍历所有过去的模式
        for past_pattern_id in past_pattern_ids:
            past_pattern_df = result_df[result_df['pattern_index'] == past_pattern_id]
            
            # 存储当前模式与这个过去模式在各个维度上的DTW距离
            individual_dtw_distances = []

            for col in dtw_columns:
                # 提取两个模式的时间序列
                series1 = current_pattern_df[col].values
                series2 = past_pattern_df[col].values

                # 计算DTW距离
                alignment = dtw(series1, series2, distance_only=True)
                distance = alignment.distance
                individual_dtw_distances.append(distance)
                distances_per_column[col].append(distance)

            # 计算五个维度的平均DTW距离
            if individual_dtw_distances:
                avg_distance = sum(individual_dtw_distances) / len(individual_dtw_distances)
                avg_distances_to_past.append(avg_distance)

        # 存储当前模式的分析结果
        dtw_results_dict[current_pattern_id] = {
            'code': current_code,
            'date': current_end_date,
            'avg_dtw_distances': avg_distances_to_past,
            'dtw_distances': distances_per_column
        }
        
    return dtw_results_dict

def save_patterns_by_date(df: pd.DataFrame, output_path: str):
    """
    将 pattern_df 按照每个模式的结束日期进行分组，
    并将每个日期的所有模式数据保存到 Excel 文件的不同工作表中。

    Args:
        df (pd.DataFrame): 包含模式数据的 DataFrame，需要有 'pattern_index' 和 'date' 列。
        output_path (str): 输出的 Excel 文件路径。
    """
    print(f"正在将结果按日期保存到 {output_path}...")
    
    # 确保日期列是datetime类型
    df['date'] = pd.to_datetime(df['date'])

    # 找到每个 pattern_index 的结束日期
    end_dates = df.groupby('pattern_index')['date'].max()
    
    # 将结束日期映射回原始 DataFrame
    df['end_date'] = df['pattern_index'].map(end_dates)
    
    # 按结束日期分组
    grouped = df.groupby('end_date')
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for date, group in tqdm(grouped, desc="Saving patterns by date"):
            # 将日期格式化为 YYYY-MM-DD 字符串，作为工作表名称
            sheet_name = date.strftime('%Y-%m-%d')
            # 将 group 数据写入对应的工作表
            group.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"结果已成功保存到 {output_path}")

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

if __name__ == "__main__":
    from prepare import get_stock_merge_table
    df = get_stock_merge_table(length=220, freq='daily')
    pattern_df = find_pattern_v1(df, window=20)
    
    # 新增调用：将结果按日期保存到数据库
    save_patterns_to_sql(pattern_df.copy(), 'patterns.db')

    # dtw_results = analyze_pattern_dtw(pattern_df)

    # # 存储结果到文件
    # # Convert keys to string because json cannot handle int64 keys
    # dtw_results_str_keys = {str(k): v for k, v in dtw_results.items()}
    # with open('dtw_results.json', 'w') as f:
    #     json.dump(dtw_results_str_keys, f, indent=4)

    # def total_sorted_results():
    #     sorted_results = sorted(dtw_results.items(), key=lambda item: len(item[1]['avg_dtw_distances']))

    #     mean_avg_distances = []
    #     for pattern_id, data in sorted_results:
    #         mean_val = np.mean(data['avg_dtw_distances'])
    #         mean_avg_distances.append(mean_val)

    #     # 3. 绘制折线图
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(range(len(mean_avg_distances)), mean_avg_distances, marker='o', linestyle='-')
                
    #     plt.title('Average DTW Distance of Patterns (Sorted by Number of Past Comparisons)')
    #     plt.xlabel('Pattern Index (Sorted by Ascending Number of Past Patterns)')
    #     plt.ylabel('Mean of Average DTW Distances')
    #     plt.grid(True)
    #     plt.show()

    # total_sorted_results()