import baostock as bs

from infra import fetch_stock_list, bs_login_required
lg = bs.login()
_,_,code, name = fetch_stock_list(end_date='2025-09-29')
print(code)
print(name)
name.to_excel('index_name.xlsx')
# rs = bs.query_all_stock('2025-9-22')
# print(rs.get_data())

# rs = bs.query_history_k_data_plus('sh.000001', "date,open,high,low,close,volume,turn,amount,pctChg", start_date='2023-01-01', end_date='2023-12-31', frequency="d", adjustflag='2')
# rs = rs.get_data()
# print(rs)   
lg = bs.logout()