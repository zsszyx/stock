import pandas as pd
from stock.sql_op.op import SqlOp
from stock.sql_op import sql_config

class DataRepository:
    """
    系统唯一的数据 IO 访问层。
    """
    def __init__(self):
        self.sql_op = SqlOp()

    def load_minutes5(self, start_date: str, end_date: str, codes: list = None) -> pd.DataFrame:
        query = f"""
            SELECT code, date, time, open, high, low, close, volume, amount 
            FROM {sql_config.mintues5_table_name}
            WHERE date >= '{start_date}' AND date <= '{end_date}'
        """
        if codes:
            codes_str = "'" + "','".join(codes) + "'"
            query += f" AND code IN ({codes_str})"
            
        df = self.sql_op.query(query)
        if df is None or df.empty:
            return pd.DataFrame()

        df['datetime'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M%S%f')
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
