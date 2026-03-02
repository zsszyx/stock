import pandas as pd
import numpy as np
import baostock as bs
from typing import List, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from stock.tasks.base import BaseTask
from stock.database.factory import RepositoryFactory
from stock.config import settings
from stock.data_fetch.data_provider.baostock_provider import BaoInterface

class NormalDailyUpdateTask(BaseTask):
    """
    常规日线数据更新任务
    职责：抓取 OHLC + Volume + Amount + Turn，并计算市值因子 (Mktcap, Rank, Pct)
    """
    def __init__(self, chunk_size: int = 100, max_workers: int = 8):
        super().__init__("NormalDailyUpdateTask")
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.repo = RepositoryFactory.get_clickhouse_repo()

    def run(self, start_date: str, end_date: str):
        self.log_progress(f"🚀 开始常规日线更新: {start_date} -> {end_date}")
        self.repo.create_normal_daily_table()

        # 1. 获取标的列表
        with BaoInterface() as bi:
            trade_dates = bi.get_trade_dates(start_date=start_date, end_date=end_date)
            if trade_dates.empty:
                self.log_progress("⚠️ 指定范围内无交易日。")
                return
            last_day = trade_dates.iloc[-1]['calendar_date']
            stock_list = bi.get_stock_list(date=last_day)
            stock_list = stock_list[stock_list['code'].str.match(r'^(sh\.60|sz\.00|sz\.30)')]
            codes = stock_list['code'].tolist()

        self.log_progress(f"📊 共有 {len(codes)} 只个股待处理...")

        # 2. 将个股列表分块
        # 虽然我们现在有了持久化会话，但分块处理依然有助于进度条展示和结果合并
        stock_chunks = [codes[i:i + 50] for i in range(0, len(codes), 50)]
        
        all_results = []
        # 使用 initializer 在每个子进程启动时仅登录一次
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=BaoInterface.worker_init
        ) as executor:
            futures = {executor.submit(self._fetch_stock_chunk, chunk, start_date, end_date): i for i, chunk in enumerate(stock_chunks)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Daily Data (Persistent Session)"):
                res_list = future.result()
                if res_list:
                    all_results.extend(res_list)

        if not all_results:
            self.log_progress("❌ 未获取到任何新数据。")
            return

        full_df = pd.concat(all_results)
        
        # 3. 计算市值因子
        print("🧮 正在计算市值因子与截面排名...")
        full_df = self._calculate_mktcap_factors(full_df)

        # 4. 写入数据库
        print(f"📤 正在写入 {len(full_df)} 行数据到 {settings.TABLE_NORMAL_DAILY}...")
        # 确保类型正确
        full_df['volume'] = full_df['volume'].astype(int)
        full_df['mktcap_rank'] = full_df['mktcap_rank'].fillna(5000).astype(int)
        
        self.repo.insert_df(full_df, settings.TABLE_NORMAL_DAILY)
        self.repo.optimize_table(settings.TABLE_NORMAL_DAILY)
        self.log_progress("🏁 常规日线更新完成。")

    def _fetch_stock_chunk(self, chunk: List[str], start_date: str, end_date: str) -> List[pd.DataFrame]:
        """
        抓取一批股票的日线数据 (直接使用预初始化的 Session)
        """
        results = []
        bi = BaoInterface()
        for code in chunk:
            try:
                df = bi.get_k_data_daily(code, start_date, end_date)
                if not df.empty:
                    results.append(df)
            except Exception:
                continue
        return results

    def _calculate_mktcap_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算市值因子：Mktcap = Close * (Volume / (Turn/100))
        并进行截面排名
        """
        # 1. 计算总市值
        # BaoStock turn 是百分比，例如 1.5 表示 1.5%
        # Total_Shares = Volume / (Turn/100)
        # 如果 turn 为 0（停牌或异常），则市值设为 NaN 随后 ffill
        df['mktcap'] = np.where(
            df['turn'] > 0,
            df['close'] * (df['volume'] / (df['turn'] / 100.0)),
            np.nan
        )
        
        # 2. 个股内填充市值（处理停牌日）
        df = df.sort_values(['code', 'date'])
        df['mktcap'] = df.groupby('code')['mktcap'].ffill().bfill()
        
        # 3. 截面排名 (按日期)
        # 排名 1 为市值最大
        df['mktcap_rank'] = df.groupby('date')['mktcap'].rank(ascending=False, method='min')
        
        # 4. 百分比排名 (0.0 到 1.0，1.0 为最大市值百分位)
        # 公式：(Rank_desc / Count) 
        # 为了符合习惯，我们计算市值领先百分比
        df['mktcap_pct'] = df.groupby('date')['mktcap'].rank(pct=True)
        
        return df

    def close(self):
        self.repo.close()
