import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from stock.tasks.base import BaseTask
from stock.database.factory import RepositoryFactory
from stock.data_fetch.data_provider.baostock_provider import BaoInterface
from stock.config import settings

worker_bi = None

def init_worker():
    global worker_bi
    worker_bi = BaoInterface()
    worker_bi._login()

def fetch_stock_turn(code, start_date, end_date):
    global worker_bi
    try:
        if worker_bi is None: init_worker()
        df = worker_bi.get_k_data_daily(code=code, start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            df['turn'] = pd.to_numeric(df['turn'], errors='coerce').fillna(0.0)
            return df[['date', 'code', 'turn']]
    except Exception: pass
    return None

class UpdateTurnoverTask(BaseTask):
    def __init__(self, max_workers: int = 10):
        super().__init__("UpdateTurnoverTask")
        self.repo = RepositoryFactory.get_clickhouse_repo()
        self.max_workers = max_workers

    def run(self):
        self.log_progress("Starting Daily Turn Update Task (Safe Mode)...")
        
        # 1. 获取全量日线数据 (用于在内存中合并)
        self.log_progress("Reading current daily table into memory...")
        df_daily = self.repo.query(f"SELECT * FROM {settings.TABLE_DAILY} ORDER BY date ASC")
        if df_daily.empty:
            self.log_progress("Daily table is empty. Run update-daily first.")
            return
            
        all_dates = sorted(df_daily['date'].unique())
        start_date, end_date = all_dates[0], all_dates[-1]
        all_codes = df_daily['code'].unique().tolist()
        self.log_progress(f"Processing {len(all_codes)} stocks.")

        # 2. 并行抓取换手率
        all_turn_data = []
        with ProcessPoolExecutor(max_workers=self.max_workers, initializer=init_worker) as executor:
            futures = [executor.submit(fetch_stock_turn, code, start_date, end_date) for code in all_codes]
            with tqdm(total=len(all_codes), desc="Fetching Turn") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res is not None:
                        all_turn_data.append(res)
                    pbar.update(1)

        if not all_turn_data:
            self.log_progress("No turnover data fetched.")
            return

        # 3. 内存合并
        self.log_progress("Merging turnover data...")
        df_turn_all = pd.concat(all_turn_data, ignore_index=True)
        
        # 移除旧的 turn 列（如果存在）并合并新的
        if 'turn' in df_daily.columns:
            df_daily = df_daily.drop(columns=['turn'])
        
        df_final = pd.merge(df_daily, df_turn_all, on=['date', 'code'], how='left')
        df_final['turn'] = df_final['turn'].fillna(0.0)

        # 4. 全量写回 (使用 TRUNCATE + INSERT 确保数据完整性)
        self.log_progress(f"Writing {len(df_final)} rows back to ClickHouse...")
        self.repo.execute(f"TRUNCATE TABLE {settings.TABLE_DAILY}")
        
        # 分块插入
        chunk_size = 100000
        for i in range(0, len(df_final), chunk_size):
            chunk = df_final.iloc[i : i + chunk_size]
            # 确保 volume 类型
            if 'volume' in chunk.columns:
                chunk['volume'] = chunk['volume'].astype(int)
            self.repo.insert_df(chunk, settings.TABLE_DAILY)

        self.repo.optimize_table(settings.TABLE_DAILY)
        self.log_progress("Turnover update completed successfully.")

    def close(self):
        self.repo.close()
