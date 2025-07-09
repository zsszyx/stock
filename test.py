import os
import requests
from init import *
from strategy import *
import efinance as ef
# 全局反扒 user-agent 设置
import akshare as ak

# print(erbo_main_query_mode())
stock_zh_a_daily_qfq_df = ak.stock_zh_a_daily(symbol="sz000001", start_date="19910403", end_date="20231027", adjust="qfq")
print(stock_zh_a_daily_qfq_df)