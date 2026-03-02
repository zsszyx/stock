import baostock as bs
import pandas as pd

from stock.utils.data_utils import DataUtils

class BaoInterface:
    _is_logged_in = False

    def __enter__(self):
        self._login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 在 Context Manager 模式下，我们不再主动登出，保持连接直到进程结束
        # 或者可以选择只在非多进程环境下登出。为了极致稳定，这里保持连接。
        pass

    def _login(self):
        if not BaoInterface._is_logged_in:
            res = bs.login()
            if res.error_code == '0':
                BaoInterface._is_logged_in = True

    def _logout(self):
        if BaoInterface._is_logged_in:
            try:
                bs.logout()
            except Exception:
                pass # 静默处理登出异常
            BaoInterface._is_logged_in = False

    @classmethod
    def worker_init(cls):
        """用于 ProcessPoolExecutor 的初始化函数"""
        instance = cls()
        instance._login()

    @classmethod
    def worker_cleanup(cls):
        """用于进程结束时的清理"""
        instance = cls()
        instance._logout()

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
