import baostock as bs
import pandas as pd
from .base import BaseConnector
from ...common.enums import Fields, FIELD_DTYPES

class BaoStockConnector(BaseConnector):
    """
    用于连接和从 BaoStock API 获取数据的连接器。
    """
    def __init__(self):
        self.bs = None
        print("BaoStockConnector 已创建。请在 with 语句中使用以确保登录和登出。")

    def __enter__(self):
        """ 上下文管理器进入时, 登录 baostock """
        print("正在登录 BaoStock...")
        self.bs = bs
        self.bs.login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ 上下文管理器退出时, 登出 baostock """
        if self.bs:
            self.bs.logout()
            print("已登出 BaoStock。")


    def get_stock_list(self, date: str | None = None) -> pd.DataFrame:
        """
        获取指定日期的所有股票列表, 如果不指定日期, 则获取最新列表。

        :param date: 查询日期, 格式 YYYY-MM-DD。如果为 None, 获取最新数据。
        :return: 一个包含股票代码、交易状态和股票名称的 DataFrame。
        """
        if not self.bs:
            raise ConnectionError("BaoStock 未连接。请在 with 语句中使用该连接器。")

        print(f"正在获取 {date or '最新'} 的股票列表...")
        rs = self.bs.query_all_stock(day=date)
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            raise ValueError(f"无法获取 {date or '最新'} 的股票列表。错误代码: {rs.error_code}, 错误信息: {rs.error_msg}")

        df = pd.DataFrame(data_list, columns=rs.fields)
        print("股票列表获取成功。")
        return df

    def get_trade_dates(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指定范围内的所有交易日。

        :param start_date: 开始日期, 格式 YYYY-MM-DD。
        :param end_date: 结束日期, 格式 YYYY-MM-DD。
        :return: 一个包含交易日和是否交易的 DataFrame。
        """
        if not self.bs:
            raise ConnectionError("BaoStock 未连接。请在 with 语句中使用该连接器。")

        print(f"正在获取从 {start_date} 到 {end_date} 的交易日历...")
        rs = self.bs.query_trade_dates(start_date=start_date, end_date=end_date)
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            raise ValueError(f"无法获取从 {start_date} 到 {end_date} 的交易日历。错误代码: {rs.error_code}, 错误信息: {rs.error_msg}")

        df = pd.DataFrame(data_list, columns=rs.fields)
        # 将 is_trading_day 列转换为整数 (0 或 1)
        df['is_trading_day'] = df['is_trading_day'].astype(int)
        print("交易日历获取成功。")
        return df

    def get_kline_data(self, code: str, start_date: str, end_date: str, frequency: str = "5", adjust_flag: str = "3") -> pd.DataFrame:
        """
        获取K线数据。
        frequency: 数据类型，默认为"5"分钟；"d"日k线,"w"周, "m"月, "5" "15" "30" "60"分钟k线
        adjust_flag: 复权类型，默认"3"为不复权；"1"前复权,"2"后复权
        """
        if not self.bs:
            raise ConnectionError("BaoStock 未连接。请在 with 语句中使用该连接器。")

        print(f"正在获取 {code} 从 {start_date} 到 {end_date} 的K线数据 (频率: {frequency})...")
        rs = self.bs.query_history_k_data_plus(
            code,
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjust_flag
        )
        if rs.error_code != '0':
            raise ValueError(f"获取K线数据失败 {code}: {rs.error_msg}")
        
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
            
        if not data_list:
            print(f"警告: 未找到 {code} 在 {start_date} 到 {end_date} 期间的数据。")
            return pd.DataFrame()

        result = pd.DataFrame(data_list, columns=rs.fields)
        
        # 1. 标准化时间列
        if frequency in ['d', 'w', 'm']:
            result[Fields.DT.value] = pd.to_datetime(result['date'])
        else:  # 分钟线
            result[Fields.DT.value] = pd.to_datetime(result['time'], format='%Y%m%d%H%M%S%f')
        
        # 2. 重命名列以符合内部标准
        result.rename(columns={'code': Fields.SYMBOL.value}, inplace=True)

        # 3. 准备数据类型映射
        dtype_map = {field.value: dtype for field, dtype in FIELD_DTYPES.items()}

        # 4. 筛选出我们需要的标准列
        standard_cols_present = [field.value for field in Fields if field.value in result.columns]
        result = result[standard_cols_present]

        # 5. 应用标准数据类型
        final_dtype_map = {col: dtype_map[col] for col in result.columns if col in dtype_map}
        result = result.astype(final_dtype_map)

        # 6. 设置索引
        result.set_index(Fields.DT.value, inplace=True)
        
        print(f"成功获取并标准化 {len(result)} 条 {code} 的K线数据。")
        return result