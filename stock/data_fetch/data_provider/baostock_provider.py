import baostock as bs
import pandas as pd

from stock.utils.data_utils import DataUtils

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

    def _process_rs(self, rs, numeric_cols: list = None) -> pd.DataFrame:
        if rs is None or rs.error_code != '0': return pd.DataFrame()
        data_list = []
        while rs.next(): data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
        if numeric_cols:
            df = DataUtils.clean_numeric_df(df, numeric_cols)
        return df

    def get_trade_dates(self, start_date: str, end_date: str) -> pd.DataFrame:
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        df = self._process_rs(rs)
        if not df.empty:
            df['is_trading_day'] = df['is_trading_day'].astype(int)
            df = df[df['is_trading_day'] == 1]
        return df

    def get_stock_list(self, date: str) -> pd.DataFrame:
        rs = bs.query_all_stock(day=date)
        return self._process_rs(rs)

    def get_k_data_5min(self, code: str, start_date: str, end_date: str, adjustflag: str = "3") -> pd.DataFrame:
        fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, end_date=end_date, frequency="5", adjustflag=adjustflag)
        return self._process_rs(rs, numeric_cols=['open', 'high', 'low', 'close', 'volume', 'amount'])

    def get_k_data_daily(self, code: str, start_date: str, end_date: str, adjustflag: str = "3") -> pd.DataFrame:
        fields = "date,code,open,high,low,close,volume,amount,turn"
        rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, end_date=end_date, frequency="d", adjustflag=adjustflag)
        return self._process_rs(rs, numeric_cols=['open', 'high', 'low', 'close', 'volume', 'amount', 'turn'])
