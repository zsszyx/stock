import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from bao.bao_interface import BaoInterface
from sql_op.op import SqlOp
from sql_op import sql_config
from tqdm import tqdm

class UpdateTask:
    def __init__(self):
        self.sql_op = SqlOp()

    def run_init_mintues5_task(self, date: str, start_date: str, end_date: str):
        """
        初始化5分钟K线数据任务
        """
        with BaoInterface() as bi:
            # 获取股票列表
            trade_dates = bi.get_trade_dates(start_date=start_date, end_date=end_date)
            last_day = trade_dates.iloc[-1]['calendar_date']
            stock_list = bi.get_stock_list(date=last_day)
            if stock_list is None:
                raise ValueError("Failed to get stock list")

            # 筛选出非指数的股票
            stock_list = stock_list[~stock_list['code'].str.contains('sh.000|sz.399')]

            # 分批处理
            batch_size = 10  # 每批处理10个股票
            for i in tqdm(range(0, len(stock_list), batch_size)):
                batch_stocks = stock_list.iloc[i:i+batch_size]
                k_data_list = []

                # 获取这批股票的最大日期
                codes_to_query = batch_stocks['code'].tolist()
                max_dates = self.sql_op.get_max_date_for_codes(codes_to_query, sql_config.mintues5_table_name)

                for index, row in tqdm(batch_stocks.iterrows(), total=batch_stocks.shape[0], desc=f"Processing batch {i//batch_size + 1}"):
                    code = row['code']
                    # 默认从start_date开始
                    current_start_date = start_date
                    if code in max_dates and max_dates[code] is not None:
                        # 如果数据库中已有数据，从最大日期当天开始获取，以补充当天可能不完整的数据
                        latest_date_in_db = pd.to_datetime(max_dates[code])
                        if latest_date_in_db.strftime('%Y-%m-%d') >= last_day:
                            # 如果已经是最新,则跳过
                            continue
                        # upsert逻辑将处理重复问题
                        current_start_date = latest_date_in_db.strftime('%Y-%m-%d')

                    k_data = bi.get_k_data_5min(code=code, start_date=current_start_date, end_date=end_date)
                    if k_data is not None and not k_data.empty:
                        k_data_list.append(k_data)

                if not k_data_list:
                    continue

                all_k_data = pd.concat(k_data_list, ignore_index=True)

                # 保存到数据库
                if not all_k_data.empty:
                    all_k_data.set_index(['code', 'time'], inplace=True)
                    self.sql_op.upsert_df_to_db(all_k_data, sql_config.mintues5_table_name, index=True)

if __name__ == '__main__':
    task = UpdateTask()
    task.run_init_mintues5_task(date='2025-12-01', start_date='2025-12-01', end_date='2026-1-1')