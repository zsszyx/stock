import baostock as bs
lg = bs.login()
rs = bs.query_all_stock('2025-9-22')
print(rs.get_data())
lg = bs.logout()