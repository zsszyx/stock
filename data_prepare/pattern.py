import pandas as pd
from tqdm import tqdm

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
    pattern_id = 1
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

    return result_df


if __name__ == "__main__":
    from prepare import get_stock_merge_table
    df = get_stock_merge_table(length=220, freq='daily')
    pattern_df = find_pattern_v1(df, window=20)
    print(pattern_df.head(40))
    print(pattern_df.info(verbose=True))
    pattern_df.describe().to_excel("pattern_summary.xlsx")