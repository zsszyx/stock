import pandas as pd
import backtrader as bt
from typing import Optional
from .bt_backtester import KSPPandasData

class BTDataFeedFactory:
    """
    Backtrader 数据馈送工厂：统一数据转换、对齐与 Fail-Fast 校验
    """
    @staticmethod
    def create_stock_feed(df: pd.DataFrame, code: str, full_idx: pd.Index) -> Optional[KSPPandasData]:
        try:
            if df.empty: return None
            
            # 1. 预处理
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['date'])
            df['is_listed'] = 1.0 # 原始数据标记为已上市
            
            # 2. 对齐时间轴 (不使用全局 bfill 以防未来函数)
            aligned = df.set_index('datetime').reindex(full_idx)
            
            # 3. 填充逻辑
            # 成交量/上市标记：缺失部分填 0
            aligned['volume'] = aligned['volume'].fillna(0.0)
            aligned['is_listed'] = aligned['is_listed'].fillna(0.0)
            
            # 价格：ffill (处理停牌), bfill (处理 Backtrader 启动)
            # 注意：即使 bfill 了价格，只要 is_listed=0，策略就不会交易
            price_cols = ['open', 'high', 'low', 'close', 'poc']
            aligned[price_cols] = aligned[price_cols].ffill().bfill().fillna(0.0)
            
            # 因子填充：
            # ffill 用于处理已上市期间的缺失值
            # 对于上市前的缺失值 (Backfilling)，使用“极其恶劣”的固定值，而非 bfill 未来值
            factor_cols = ['ksp_rank', 'ksp_sum_14d_rank', 'ksp_sum_10d_rank', 'ksp_sum_7d_rank', 'ksp_sum_5d_rank', 'list_days', 'is_listed_180', 'ksp_sum_5d', 'ksp_sum_10d']
            for col in factor_cols:
                if col in aligned.columns:
                    # 分两步：1. ffill 处理空洞；2. 用安全值填充上市前的部分
                    aligned[col] = aligned[col].ffill()
                    if 'rank' in col:
                        aligned[col] = aligned[col].fillna(5000.0) # 排名垫底 (全市场约 5000 只票)
                    else:
                        aligned[col] = aligned[col].fillna(0.0)
                else:
                    if 'rank' in col:
                        aligned[col] = 5000.0
                    else:
                        aligned[col] = 0.0
            
            # 4. 构建 Final DataFrame
            final_df = aligned[['open', 'high', 'low', 'close', 'volume']].copy()
            final_df['openinterest'] = 0.0
            for col in factor_cols:
                final_df[col] = aligned[col]
            final_df['poc_line'] = aligned['poc']
            final_df['is_listed'] = aligned['is_listed']
            
            final_df = final_df.astype(float)
            
            # 5. 创建 DataFeed
            return KSPPandasData(
                dataname=final_df,
                datetime=None,
                open=0, high=1, low=2, close=3, volume=4, openinterest=5,
                ksp_rank=6, ksp_sum_14d_rank=7, ksp_sum_10d_rank=8, ksp_sum_7d_rank=9, ksp_sum_5d_rank=10,
                list_days=11, is_listed_180=12, ksp_sum_5d=13, ksp_sum_10d=14, poc=15, is_listed=16,
                name=code, plot=False
            )
        except Exception:
            return None

    @staticmethod
    def create_benchmark_feed(df: pd.DataFrame, full_idx: pd.Index) -> bt.feeds.PandasData:
        b_df = df.copy()
        b_df['datetime'] = pd.to_datetime(b_df['date'])
        # 基准数据允许 bfill 以对齐初始时间
        b_df = b_df.set_index('datetime').reindex(full_idx).ffill().bfill()
        feed_df = b_df[['close']].astype(float)
        return bt.feeds.PandasData(dataname=feed_df, name='_master_clock_', plot=False)
