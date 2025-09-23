import baostock as bs
lg = bs.login()
rs = bs.query_all_stock('2025-9-22')
print(rs.get_data())

rs = bs.query_history_k_data_plus('sh.000001', "date,open,high,low,close,volume,turn,amount,pctChg", start_date='2023-01-01', end_date='2023-12-31', frequency="d", adjustflag='2')
rs = rs.get_data()
print(rs)   
lg = bs.logout()