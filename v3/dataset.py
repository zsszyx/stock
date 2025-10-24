import numba
import numpy as np
import pandas as pd
from prepare import get_stock_merge_industry_table
from metric import atr

class ShortLabelGenerator:
    """
    短期标签生成器。
    
    标签定义:
    - 2: 止盈 (价格触及上方障碍)
    - 1: 超时 (持有至时间限制仍未触及任何障碍)
    - 0: 止损 (价格触及下方障碍)
    - NaN: 未来数据不足，无法判断
    
    参数:
    - time_limit: 最大持有天数
    - tp_mult: 止盈障碍的ATR倍数
    - sl_mult: 止损障碍的ATR倍数
    """
    def __init__(self, time_limit: int, tp_mult: float, sl_mult: float):
        self.time_limit = time_limit
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult

    @staticmethod
    @numba.jit(nopython=True)
    def compute_labels_numba(highs: np.ndarray, 
                             lows: np.ndarray, 
                             closes: np.ndarray, 
                             atr14s: np.ndarray, 
                             time_limit: int, 
                             tp_mult: float, 
                             sl_mult: float):
        """
        Numba JIT编译的核心逻辑。
        标签: 2=止盈, 1=超时, 0=止损, np.nan=未来数据不足
        """
        num_rows = len(closes)
        labels = np.full(num_rows, 1.0) 
        
        for i in range(num_rows - time_limit):
            top_barrier = closes[i] + (atr14s[i] * tp_mult)
            bottom_barrier = closes[i] - (atr14s[i] * sl_mult)
            
            for j in range(1, time_limit + 1):
                future_idx = i + j
                
                if lows[future_idx] <= bottom_barrier:
                    labels[i] = 0.0 
                    break 
                
                if highs[future_idx] >= top_barrier:
                    labels[i] = 2.0 
                    break 
            
        labels[num_rows - time_limit:] = np.nan
        return labels

    def generate_labels_for_group(self,
                                  df_group: pd.DataFrame, 
                                  time_limit: int, 
                                  tp_mult: float, 
                                  sl_mult: float) -> pd.Series:
            """
            为 groupby().apply() 准备的包装函数
            """
            highs = df_group['high'].values
            lows = df_group['low'].values
            closes = df_group['close'].values
            atr14s = df_group['atr14'].values
            
            labels_array = self.compute_labels_numba(
                highs, lows, closes, atr14s, 
                time_limit, tp_mult, sl_mult
            )
            
            return pd.Series(labels_array, index=df_group.index)
    
    def main(self,
             df: pd.DataFrame) -> pd.Series:
        """
        主函数：为整个DataFrame生成短期标签
        """
        print("\n开始生成标签...")
        labels = df.groupby(level='code').apply(
            self.generate_labels_for_group,
            time_limit=self.time_limit,
            tp_mult=self.tp_mult,
            sl_mult=self.sl_mult
        )
        
        labels = labels.reset_index(level=0, drop=True)
        df['short_term_label'] = labels
        self.result = df
        return df

    def analyze_labels(self):
        """
        分析标签的分布情况
        """
        label_counts = self.result['short_term_label'].value_counts(dropna=False)
        print("\n标签分布情况:")
        print(label_counts)

    def validate_labels(self):
        """
        验证标签的正确性（抽样检查）
        """
        sample = self.result[['high', 'low', 'close', 'atr14', 'short_term_label']].dropna().sample(5)
        print("\n--- 标签验证抽样 ---")
        for idx, row in sample.iterrows():
            code, date = idx
            close_price = row['close']
            atr_val = row['atr14']
            label = row['short_term_label']
            
            top_barrier = close_price + (atr_val * self.tp_mult)
            bottom_barrier = close_price - (atr_val * self.sl_mult)
            
            print(f"\n--- 验证样本: Code={code}, Date={date} ---")
            print(f"标签: {label} (0=止损, 1=超时, 2=止盈)")
            print(f"当日收盘价: {close_price:.2f}, ATR: {atr_val:.2f}")
            print(f"止盈障碍 > {top_barrier:.2f}")
            print(f"止损障碍 < {bottom_barrier:.2f}")
            
            stock_data = self.result.loc[code]
            future_data = stock_data.loc[date:].iloc[1:self.time_limit + 1]
            
            print("未来收益情况:")
            if not future_data.empty:
                high_returns = (future_data['high'] - close_price) / close_price * 100
                low_returns = (future_data['low'] - close_price) / close_price * 100
                
                max_profit = high_returns.max()
                max_loss = low_returns.min()

                for i, (f_date, f_row) in enumerate(future_data.iterrows()):
                    day_num = i + 1
                    high_price = f_row['high']
                    low_price = f_row['low']
                    
                    hit_top = "<- 触及止盈" if high_price >= top_barrier else ""
                    hit_bottom = "<- 触及止损" if low_price <= bottom_barrier else ""
                    
                    print(f"  T+{day_num:02d} ({f_date}): "
                          f"High={high_price:.2f} ({high_returns.iloc[i]:+6.2f}%), "
                          f"Low={low_price:.2f} ({low_returns.iloc[i]:+6.2f}%) "
                          f"{hit_top}{hit_bottom}")

                print(f"\n  期间最大潜在盈利: {max_profit:+.2f}%")
                print(f"  期间最大潜在亏损: {max_loss:+.2f}%")
            else:
                print("  无未来数据可供分析。")

class RollingWindowSplitter:
    """
    一个用于前向滚动回测的提取器类。
    
    它为具有MultiIndex (code, date)的DataFrame生成
    (训练集, 测试集) 的数据对。
    """
    
    def __init__(self, train_window_size: str, test_window_size: str):
        """
        初始化提取器。
        
        参数:
        train_window_size (str): 训练窗口的大小 (例如 '5Y' - 5年, '36M' - 36个月)。
        test_window_size (str): 测试窗口的大小，也充当滚动的“步长” 
                                 (例如 '3M' - 3个月, '1Q' - 1个季度)。
        """
        try:
            self.train_offset = pd.tseries.frequencies.to_offset(train_window_size)
            self.test_offset = pd.tseries.frequencies.to_offset(test_window_size)
        except ValueError as e:
            raise ValueError(f"窗口大小格式错误。请使用Pandas offset字符串 (例如 '5Y', '6M')。错误: {e}")
            
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self._all_dates_cache = None

    def _get_all_dates(self, df: pd.DataFrame) -> pd.Series:
        """
        一个辅助方法，用于提取、排序并缓存DataFrame中所有唯一的日期。
        """
        all_dates = pd.to_datetime(df.index.get_level_values('date').unique()).sort_values()
        
        if all_dates.empty:
            raise ValueError("DataFrame 在 'date' 索引层中不包含任何日期。")
            
        self._all_dates_cache = all_dates
        return self._all_dates_cache

    def get_window_boundaries(self, df: pd.DataFrame):
        """
        一个生成器，仅 yield 每个窗口的 (train_start, train_end, test_end) 日期边界。
        
        这对于调试或可视化窗口划分非常有用。
        """
        all_dates = self._get_all_dates(df)
        data_start_date = all_dates.min()
        data_end_date = all_dates.max()

        current_train_start = data_start_date
        current_train_end = current_train_start + self.train_offset
        current_test_end = current_train_end + self.test_offset
        
        while current_test_end <= data_end_date:
            
            yield (current_train_start, current_train_end, current_test_end)
            
            current_train_start += self.test_offset
            current_train_end += self.test_offset
            current_test_end += self.test_offset

    def get_n_splits(self, df: pd.DataFrame) -> int:
        """
        返回此提取器将为给定DataFrame生成的总切片数量。
        (类似于 scikit-learn 的 CV splitters)
        """
        return sum(1 for _ in self.get_window_boundaries(df))

    def split(self, df: pd.DataFrame):
        """
        核心方法：一个生成器，用于切分DataFrame并 yield (train_data, test_data) 对。
        
        参数:
        df (pd.DataFrame): 必须有MultiIndex (code, date) 且已排序。
        
        Yields:
        tuple (pd.DataFrame, pd.DataFrame): (训练集, 测试集) 的数据对。
        """
        
        date_index = df.index.get_level_values('date')
        
        for train_start, train_end, test_end in self.get_window_boundaries(df):
            
            train_mask = (date_index >= train_start) & (date_index < train_end)
            
            test_mask = (date_index >= train_end) & (date_index < test_end)
            
            train_data = df[train_mask]
            test_data = df[test_mask]
            
            if not train_data.empty and not test_data.empty:
                yield train_data, test_data

class RollingWindowSplitter:
    def __init__(self, train_window_size, test_window_size):
        self.train_offset = pd.tseries.frequencies.to_offset(train_window_size)
        self.test_offset = pd.tseries.frequencies.to_offset(test_window_size)
    def get_window_boundaries(self, df):
        all_dates = pd.to_datetime(df.index.get_level_values('date').unique()).sort_values()
        data_start_date, data_end_date = all_dates.min(), all_dates.max()
        current_train_start = data_start_date
        current_train_end = current_train_start + self.train_offset
        current_test_end = current_train_end + self.test_offset
        while current_test_end <= data_end_date:
            yield (current_train_start, current_train_end, current_test_end)
            current_train_start += self.test_offset
            current_train_end += self.test_offset
            current_test_end += self.test_offset
    def split(self, df):
        date_index = df.index.get_level_values('date')
        for train_start, train_end, test_end in self.get_window_boundaries(df):
            train_mask = (date_index >= train_start) & (date_index < train_end)
            test_mask = (date_index >= train_end) & (date_index < test_end)
            if train_mask.any() and test_mask.any():
                yield df[train_mask], df[test_mask]

def create_dummy_data(years=7):
    total_days = int(365.25 * years)
    dates = pd.to_datetime(pd.date_range('2018-01-01', periods=total_days, freq='B'))
    codes = ['stock_A', 'stock_B', 'stock_C']
    index = pd.MultiIndex.from_product([codes, dates], names=['code', 'date'])
    df = pd.DataFrame(index=index)
    df['feature_1'] = df.groupby(level='code')['date'].transform(lambda x: np.random.randn(len(x)).cumsum())
    df['feature_2'] = df.groupby(level='code')['date'].transform(lambda x: np.random.randn(len(x)).cumsum())
    df['atr14'] = np.random.rand(len(index)) * 2 + 1
    df['short_term_label'] = np.random.choice([0.0, 1.0, 2.0, np.nan], len(index), p=[0.3, 0.4, 0.2, 0.1])
    return df.sort_index()

merged = create_dummy_data()

TRAIN_WINDOW = '3Y'  
TEST_WINDOW = '3M'   

FEATURE_COLUMNS = ['feature_1', 'feature_2', 'atr14'] 
LABEL_COLUMN = 'short_term_label'
LOOKBACK_WINDOW = 10 

LGBM_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'n_jobs': -1,
    'random_state': 42
}

splitter = RollingWindowSplitter(
    train_window_size=TRAIN_WINDOW,
    test_window_size=TEST_WINDOW
)

all_oos_predictions = []

for i, (train_df, test_df) in enumerate(splitter.split(merged)):
    
    print(f"\n--- 滚动窗口 {i+1} ---")
    train_dates = train_df.index.get_level_values('date')
    test_dates = test_df.index.get_level_values('date')
    print(f"Train: [{train_dates.min().date()} : {train_dates.max().date()}]")
    print(f"Test:  [{test_dates.min().date()} : {test_dates.max().date()}]")

    trainer = TemporalTrainer(
        model_type='lgbm',
        model_params=LGBM_PARAMS,
        feature_columns=FEATURE_COLUMNS,
        label_column=LABEL_COLUMN,
        lookback_window=LOOKBACK_WINDOW
    )
    
    trainer.fit(train_df)
    
    predictions_df = trainer.predict(
        test_df=test_df, 
        train_history_df=train_df 
    )
    
    if not predictions_df.empty:
        results = test_df[[LABEL_COLUMN]].copy()
        results = results.join(predictions_df)
        all_oos_predictions.append(results)
    else:
        print("此次滚动没有生成预测。")

print("\n--- 回测结束 ---")
if all_oos_predictions:
    final_oos_report = pd.concat(all_oos_predictions)
    
    final_oos_report_clean = final_oos_report.dropna(subset=[LABEL_COLUMN, 'prediction'])
    
    print(f"总共收集到 {len(final_oos_report_clean)} 条可评估的样本外预测。")
    print(final_oos_report_clean.head())
    
    from sklearn.metrics import classification_report
    print("\n--- 最终OOS分类报告 ---")
    print(classification_report(
        final_oos_report_clean[LABEL_COLUMN], 
        final_oos_report_clean['prediction']
    ))
else:
    print("没有生成任何预测结果。")

TIME_LIMIT = 10      
TP_MULTIPLIER = 2.0  
SL_MULTIPLIER = 1.0  

merged = get_stock_merge_industry_table(length=100, freq='daily')
merged = atr(merged, timeperiod=14)

label_generator = ShortLabelGenerator(
    time_limit=TIME_LIMIT,
    tp_mult=TP_MULTIPLIER,
    sl_mult=SL_MULTIPLIER
)
labeled_df = label_generator.main(merged)
label_generator.analyze_labels()
label_generator.validate_labels()
