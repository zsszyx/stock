import baostock as bs
import pandas as pd

class BaoInterface:
    def __enter__(self):
        self._login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._logout()

        def _login(self):
            bs.login()
    
        def _logout(self):
            bs.logout()
    
        def get_trade_dates(self, start_date: str, end_date: str) -> pd.DataFrame:
            """
            获取交易日信息
            """
            rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
            if rs.error_code != '0':
                return None
    
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            result = pd.DataFrame(data_list, columns=rs.fields)
            result['is_trading_day'] = result['is_trading_day'].astype(int)
            result = result[result['is_trading_day'] == 1]
            return result
    
        def get_stock_list(self, date: str) -> pd.DataFrame:
            """
            获取某日所有证券信息
            """
            rs = bs.query_all_stock(day=date)
            if rs.error_code != '0':
                return None
    
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            result = pd.DataFrame(data_list, columns=rs.fields)
            return result
    
        def get_k_data_5min(self, code: str, start_date: str, end_date: str, adjustflag: str = "3") -> pd.DataFrame:
            """
            获取5分钟K线数据
            adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权。
            """
            fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
            rs = bs.query_history_k_data_plus(code,
                fields,
                start_date=start_date, end_date=end_date,
                frequency="5", adjustflag=adjustflag)
            
            if rs.error_code != '0':
                return None
    
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            result = pd.DataFrame(data_list, columns=rs.fields)
            return result
    
        def get_k_data_daily(self, code: str, start_date: str, end_date: str, adjustflag: str = "3") -> pd.DataFrame:
            """
            获取日线K线数据
            """
            fields = "date,code,open,high,low,close,volume,amount,turn"
            rs = bs.query_history_k_data_plus(code,
                fields,
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag=adjustflag)
            
            if rs.error_code != '0':
                return None
    
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            result = pd.DataFrame(data_list, columns=rs.fields)
            return result
    