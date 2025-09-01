import baostock as bs
from init import DB_PATH
lg = bs.login()
rs = bs.query_stock_industry()
print(rs.get_data().to_excel('stock_industry.xlsx'))

bs.logout()

