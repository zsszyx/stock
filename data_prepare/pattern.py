import pandas as pd
from tqdm import tqdm
from dtw import dtw
from sklearn.preprocessing import StandardScaler

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

    if not len(pattern_start_indices):
        print("未找到符合条件的模式。")
        return pd.DataFrame()

    # 提取所有符合条件的20天数据段
    pattern_dfs = []
    pattern_id = 0
    for end_idx in tqdm(pattern_start_indices, desc="Extracting patterns"):
        start_idx = end_idx - window
        pattern_chunk = df.iloc[start_idx:end_idx].copy()
        pattern_chunk['pattern_index'] = pattern_id
        pattern_dfs.append(pattern_chunk)
        pattern_id += 1

    if not pattern_dfs:
        return pd.DataFrame()

    # 合并所有模式数据
    result_df = pd.concat(pattern_dfs, ignore_index=True)

    # --- DTW 计算 ---
    dtw_results = {}
    columns_to_compare = ['close', 'open', 'high', 'low', 'volume']
    scaler = StandardScaler()

    for i in tqdm(range(len(pattern_dfs)), desc="Calculating DTW"):
        pattern_i_code = pattern_dfs[i]['code'].iloc[0]
        pattern_i_date = pattern_dfs[i]['date'].iloc[-1]
        
        # 初始化当前 pattern 的 DTW 记录
        dtw_results[i] = {
            'code': pattern_i_code,
            'date': pattern_i_date,
            'dtw_distances': {col: [] for col in columns_to_compare},
            'avg_dtw_distances': []
        }

        # 只与过去的 pattern 比较
      
        for j in range(i):
            pattern_i_data = pattern_dfs[i]
            pattern_j_data = pattern_dfs[j]
                
            dtw_vals_for_pair = []
            for col in columns_to_compare:
                # 标准化
                series_i = scaler.fit_transform(pattern_i_data[[col]]).flatten()
                series_j = scaler.fit_transform(pattern_j_data[[col]]).flatten()

                # 计算DTW
                distance = dtw(series_i, series_j, keep_internals=True).distance
                dtw_results[i]['dtw_distances'][col].append(distance)
                dtw_vals_for_pair.append(distance)

            # 计算并记录均值
            if dtw_vals_for_pair:
                avg_dist = sum(dtw_vals_for_pair) / len(dtw_vals_for_pair)
                dtw_results[i]['avg_dtw_distances'].append(avg_dist)

    return result_df, dtw_results



if __name__ == "__main__":
    from prepare import get_stock_merge_table
    df = get_stock_merge_table(length=220, freq='daily')
    pattern_df, dtw_results = find_pattern_v1(df, window=20)
    
    if not pattern_df.empty:
        print(pattern_df.head(40))
        print(pattern_df.info(verbose=True))
        pattern_df.describe().to_excel("pattern_summary.xlsx")

        # 打印部分DTW结果作为演示
        if dtw_results:
            print("\n--- DTW Calculation Results ---")
            for i in range(1, min(6, len(dtw_results) + 1)):
                print(f"\nPattern {i} (Code: {dtw_results[i]['code']}, Date: {dtw_results[i]['date']}):")
                if dtw_results[i]['avg_dtw_distances']:
                    print(f"  - Average DTW distances to past patterns: {dtw_results[i]['avg_dtw_distances']}")
                    print(f"  - DTW distances for 'close' to past patterns: {dtw_results[i]['dtw_distances']['close']}")
                else:
                    print("  - No past patterns to compare.")
    else:
        print("未找到符合条件的模式。")