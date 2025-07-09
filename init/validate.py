import pandas as pd
import numpy as np
import warnings
import inspect

def check_data(df: pd.DataFrame):
    """
    通用的数据清洗流程。
    第一步：检查是否有某一行或某一列全为空值（包括空字符串、None 和 NaN），如果有则抛出异常。
    第二步：检查是否存在重复行，如果有则抛出异常。
    第三步：检查是否存在无穷大或无穷小值，如果有则抛出异常。
    如果某一列或某一行部分为空值，则抛出一个 Warning，并附加调用 check_data 的上一级函数的名称以及传入的参数。
    :param df: 待清洗的金融数据 DataFrame
    :return: 清洗后的 DataFrame
    """
    # 获取调用 check_data 的上一级函数的名称和参数
    caller_frame = inspect.stack()[1]
    caller_name = caller_frame.function
    caller_args = inspect.getargvalues(caller_frame.frame).locals

    # 将空字符串替换为 NaN，统一处理
    df.replace("", pd.NA, inplace=True)

    # 检查是否有某一行全为空值
    if df.isnull().all(axis=1).any():
        raise ValueError(f"数据中存在某些行完全为空值，请检查数据！调用函数: {caller_name}, 参数: {caller_args}")

    # 检查是否有某一列全为空值
    if df.isnull().all(axis=0).any():
        raise ValueError(f"数据中存在某些列完全为空值，请检查数据！调用函数: {caller_name}, 参数: {caller_args}")

    # 检查是否有某一行部分为空值
    if df.isnull().any(axis=1).any():
        warnings.warn(f"数据中存在某些行部分为空值。调用函数: {caller_name}, 参数: {caller_args}")

    # 检查是否有某一列部分为空值
    if df.isnull().any(axis=0).any():
        warnings.warn(f"数据中存在某些列部分为空值。调用函数: {caller_name}, 参数: {caller_args}")

    # 检查是否存在重复行
    if df.duplicated().any():
        raise ValueError(f"数据中存在重复行，请检查数据！调用函数: {caller_name}, 参数: {caller_args}")

    # 检查是否存在无穷大或无穷小值，仅针对数值列
    numeric_cols = df.select_dtypes(include=[np.number])  # 选择数值列
    if np.isinf(numeric_cols.values).any():
        raise ValueError(f"数据中存在无穷大或无穷小值，请检查数据！调用函数: {caller_name}, 参数: {caller_args}")

    return df


if __name__ == "__main__":
    # 示例数据
    data = {
        "代码": ["000001", "000002", None, "000004", "000001"],
        "名称": ["平安银行", "", None, "招商银行", "平安银行"],
        "价格": [12.34, None, np.inf, None, 12.34]
    }
    df = pd.DataFrame(data)

    try:
        cleaned_df = check_data(df)
        print("清洗后的数据：")
        print(cleaned_df)
    except ValueError as e:
        print(f"数据清洗异常: {e}")