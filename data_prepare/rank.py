import pandas as pd

class BaseFilter:
    def __init__(self, df):
        self.df = df

    def apply_filter(self, date) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement this method.")
    
class BollingFilter(BaseFilter):
    def __init__(self, df, period=20, keep=0.9):
        super().__init__(df)
        self.period = period
        self.keep = keep
        self.add_bolling()

    def add_bolling(self):
        df = self.df
        print("正在计算布林带指标...")
        period = self.period
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
 
    def apply_filter(self, date) -> pd.DataFrame:
        df_today = self.df[self.df['date'] == date].copy()
        if df_today.empty:
            raise ValueError(f"No data available for the specified date: {date}")

        # 确保用于分位数计算的列没有NaN值
        df_today.dropna(subset=['volume_bb', 'close_bb'], inplace=True)
        if df_today.empty:
            raise ValueError(f"No valid data available for the specified date: {date}")

        # 计算分位数阈值，以过滤掉最大的 (1-keep) 部分
        # 例如，如果 keep=0.9，则保留前90%的数据，即过滤掉大于90%分位数的数据
        volume_bb_threshold = df_today['volume_bb'].quantile(self.keep)
        close_bb_threshold = df_today['close_bb'].quantile(self.keep)

        # 应用过滤器，保留小于或等于阈值的股票
        filtered_df = df_today[
            (df_today['volume_bb'] <= volume_bb_threshold) &
            (df_today['close_bb'] <= close_bb_threshold)
        ]

        return filtered_df['code']

class LimitFilter(BaseFilter):
    def __init__(self, df):
        super().__init__(df)

    def apply_filter(self, date) -> pd.DataFrame:
        df_today = self.df[self.df['date'] == date].copy()
        # 过滤掉涨停或跌停的股票
        limit_up_stocks = df_today[df_today['pctChg'].abs() <= 9.85]
        return limit_up_stocks['code']

class HighCloseFilter(BaseFilter):
    def __init__(self, df):
        super().__init__(df)
        # 计算当前价格和最近20天价格的偏离程度
        ma20 = self.df['close'].rolling(window=20).mean()
        self.df['high_close_deviation'] = (self.df['close'] - ma20) / ma20

    def apply_filter(self, date) -> pd.DataFrame:
        # 过滤掉偏离程度大于0.1的股票
        df_today = self.df[self.df['date'] == date].copy()
        high_close_stocks = df_today[df_today['high_close_deviation'] <= 0.1]
        return high_close_stocks['code']

class LatestChgFilter(BaseFilter):
    def __init__(self, df, window=10):
        super().__init__(df)
        # 计算最近window天的变化率最大值是否大于5的股票
        self.df['latest_max_chg'] = self.df.groupby('code')['pctChg'].rolling(window=window, min_periods=window).max().reset_index(level=0, drop=True)

    def apply_filter(self, date) -> pd.DataFrame:
        # 过滤掉最近window天变化率最大值大于5的股票
        df_today = self.df[self.df['date'] == date].copy()
        latest_chg_stocks = df_today[df_today['latest_max_chg'] < 5]
        return latest_chg_stocks['code']
    
class CurvatureFilter(BaseFilter):
    def __init__(self, df, period=60):
        super().__init__(df)
        self.period = period
        self.add_curvature()
    
    def add_curvature(self):
        df = self.df
        # 60日均线曲率
        period = 20
        def compute_curvature(group, col='close'):
            # 计算60日均线的二阶导数作为曲率指标
            group['ma'] = group[col].rolling(window=period, min_periods=period).mean()
            group[f'curvature_{col}'] = group['ma'].diff().diff()
            return group
        df = df.groupby('code').apply(compute_curvature, col='close').reset_index(drop=True)
        period = 10
        df = df.groupby('code').apply(compute_curvature, col='volume').reset_index(drop=True)
        def check_recent_curvature(group):
            # 检查最近3天的volume和close曲率是否都大于等于0
            group['recent_curvature_positive'] = (
            (group['curvature_close'].rolling(window=2, min_periods=2).min() >= 0) &
            (group['curvature_volume'].rolling(window=2, min_periods=2).min() >= 0)
            )
            return group
        
        df = df.groupby('code').apply(check_recent_curvature).reset_index(drop=True)
        self.df = df

    def apply_filter(self, date) -> pd.DataFrame:
        df_today = self.df[self.df['date'] == date].copy()
        # 过滤掉60日均线曲率大于0的股票
        curvature_stocks = df_today[(df_today[f'recent_curvature_positive'])]
        return curvature_stocks['code']

class UpShadowFilter(BaseFilter):
    def __init__(self, df):
        super().__init__(df)
        # 计算最近三天上影线比例是否超过close的3%
        self.df['up_shadow'] = (self.df['high']) / self.df['close']
        self.df['up_shadow'] = self.df['up_shadow'].groupby(self.df['code']).rolling(window=3).max().reset_index(level=0, drop=True)

    def apply_filter(self, date) -> pd.DataFrame:
        df_today = self.df[self.df['date'] == date].copy()
        up_shadow_stocks = df_today[df_today['up_shadow'] <= 1.03]
        return up_shadow_stocks['code']

class ColunmnsAdder:
    def __init__(self, df):
        self.df = df

    def compute_mean_curvature(self, group, period, col='close'):
        # 计算60日均线的二阶导数作为曲率指标
        ma = group[col].rolling(window=period, min_periods=period).mean()
        group[f'curvature_{col}_{period}'] = ma.diff().diff()
        return group

    def compute_mean_diff(self, group, period, col='close'):
        # 计算60日均线的变化率作为指标
        ma = group[col].rolling(window=period, min_periods=period).mean()
        group[f'mean_diff_{col}_{period}'] = ma.diff()
        return group

    def up_shadow(self, group):
        group['up_shadow'] = (group['high']) / group['close']
        group['up_shadow'] = group['up_shadow'].rolling(window=3).max()
        return group
    
    def add_columns(self):
        df = self.df
        df = df.groupby('code').apply(self.compute_curvature, period=60, col='close').reset_index(drop=True)
        df = df.groupby('code').apply(self.compute_curvature, period=20, col='close').reset_index(drop=True)
        df = df.groupby('code').apply(self.compute_curvature, period=10, col='volume').reset_index(drop=True)
        df = df.groupby('code').apply(self.up_shadow).reset_index(drop=True)
        df = df.groupby('code').apply(self.compute_mean_diff, period=20, col='close').reset_index(drop=True)
        df = df.groupby('code').apply(self.compute_mean_diff, period=60, col='close').reset_index(drop=True)
        self.df = df
        return df
    
    def filter_by_date(self, date):
        df_today = self.df[self.df['date'] == date].copy()
        
        return df_today
    
if __name__ == "__main__":
    from prepare import get_stock_merge_table
    date = '2025-10-15'
    df = get_stock_merge_table(220)
    # filter = BollingFilter(df)
    # codes = filter.apply_filter(date)
    # print(codes.info())

    limit_up_filter = LimitFilter(df)
    limit_up_codes = limit_up_filter.apply_filter(date)
    print(limit_up_codes.info())
    # high_close_filter = HighCloseFilter(df)
    
    # high_close_codes = high_close_filter.apply_filter(date)
    # print(high_close_codes.info())

    latest_chg_filter = LatestChgFilter(df, window=5)
    latest_chg_codes = latest_chg_filter.apply_filter(date)
    print(latest_chg_codes.info())

    curvature_filter = CurvatureFilter(df)
    curvature_codes = curvature_filter.apply_filter(date)
    print(curvature_codes.info())
    
    up_shadow_filter = UpShadowFilter(df)
    up_shadow_codes = up_shadow_filter.apply_filter(date)
    print(up_shadow_codes.info())

    # 求交集
    final_codes = set(curvature_codes) & set(up_shadow_codes) & set(latest_chg_codes) & set(limit_up_codes)
    print(final_codes)
    print(f"最终筛选出 {len(final_codes)} 只股票")

    # 根据code和date筛选出df中的数据，并按上影线排序
    if final_codes:
        selected_stocks_df = df[(df['code'].isin(final_codes)) & (df['date'] == date)].copy()
        sorted_stocks = selected_stocks_df.sort_values(by='up_shadow', ascending=True)
        
        print("\n按上影线从短到长排序后的股票：")
        print(sorted_stocks[['code', 'up_shadow']].head(20))
    # 随机选择10只股票
    # import random
    # sample_size = min(5, len(final_codes))
    # random_codes = random.sample(list(final_codes), sample_size)
    # print(random_codes)