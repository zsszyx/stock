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
    # --- 2. 核心：Numba JIT 加速的标签计算函数 ---

    # nopython=True 确保代码完全在C语言层面运行，速度最快
    # parallel=True 可以(但在本例中未启用)并行化
    
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
        labels = np.full(num_rows, 1.0) # 默认标签 = 1.0 (超时)
        
        for i in range(num_rows - time_limit):
            # 1. 设置当天的障碍 (T日)
            top_barrier = closes[i] + (atr14s[i] * tp_mult)
            bottom_barrier = closes[i] - (atr14s[i] * sl_mult)
            
            # 2. 向前看 (T+1 到 T+time_limit)
            for j in range(1, time_limit + 1):
                future_idx = i + j
                
                # 3. 检查“最先”触碰
                # 必须先检查止损，再检查止盈 (或者反过来，但顺序必须固定)
                if lows[future_idx] <= bottom_barrier:
                    labels[i] = 0.0 # 止损
                    break # 已触碰，停止向前看
                
                if highs[future_idx] >= top_barrier:
                    labels[i] = 2.0 # 止盈
                    break # 已触碰，停止向前看
            
            # 如果内层循环(j)跑完了都没有break，标签保持为1.0
            
        # 4. 数据末尾的标签设为 NaN
        labels[num_rows - time_limit:] = np.nan
        return labels

    # --- 3. Pandas 包装器函数 ---
    def generate_labels_for_group(self,
                                  df_group: pd.DataFrame, 
                                  time_limit: int, 
                                  tp_mult: float, 
                                  sl_mult: float) -> pd.Series:
            """
            为 groupby().apply() 准备的包装函数
            """
            # 从DataFrame中提取Numpy数组
            highs = df_group['high'].values
            lows = df_group['low'].values
            closes = df_group['close'].values
            atr14s = df_group['atr14'].values
            
            # 调用高速的 numba 函数
            labels_array = self.compute_labels_numba(
                highs, lows, closes, atr14s, 
                time_limit, tp_mult, sl_mult
            )
            
            # 将结果数组包装回带有原始索引的Series
            return pd.Series(labels_array, index=df_group.index)
    
    def main(self,
             df: pd.DataFrame) -> pd.Series:
        """
        主函数：为整个DataFrame生成短期标签
        """
        print("\n开始生成标签...")
        # 按 'code' (索引的第0层) 分组，然后应用包装函数
        labels = df.groupby(level='code').apply(
            self.generate_labels_for_group,
            time_limit=self.time_limit,
            tp_mult=self.tp_mult,
            sl_mult=self.sl_mult
        )
        
        # .apply() 的结果会包含 'code' 索引，我们需要把它去掉以便对齐
        labels = labels.reset_index(level=0, drop=True)
        # 将生成的标签合并回主数据框
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
            
            # 获取该股票的未来数据
            stock_data = self.result.loc[code]
            # 使用 .loc 获取从当前日期之后的数据
            future_data = stock_data.loc[date:].iloc[1:self.time_limit + 1]
            
            print("未来收益情况:")
            if not future_data.empty:
                # 计算相对于入场价的回报率
                high_returns = (future_data['high'] - close_price) / close_price * 100
                low_returns = (future_data['low'] - close_price) / close_price * 100
                
                # 找到最大潜在盈利和亏损
                max_profit = high_returns.max()
                max_loss = low_returns.min()

                # 打印每日详情
                for i, (f_date, f_row) in enumerate(future_data.iterrows()):
                    day_num = i + 1
                    high_price = f_row['high']
                    low_price = f_row['low']
                    
                    # 检查当天是否触及障碍
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
# 定义短期标签的参数
TIME_LIMIT = 10      # 短期：最多持仓10天
TP_MULTIPLIER = 2.0  # 止盈：当前价格 + 2 * ATR
SL_MULTIPLIER = 1.0  # 止损：当前价格 - 1.5 * ATR

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
