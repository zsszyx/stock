import baostock as bs
import pandas as pd

# 设置pandas显示选项，打印所有行
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)

def download_data_sample(date):
    bs.login()

    # 获取指定日期的指数、股票数据
    stock_rs = bs.query_all_stock(date)
    stock_df = stock_rs.get_data()
    print(stock_df)

def trade_date_sample():
    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    #### 获取交易日信息 ####
    rs = bs.query_trade_dates(start_date="2017-01-01", end_date="2025-07-30")
    print('query_trade_dates respond error_code:'+rs.error_code)
    print('query_trade_dates respond  error_msg:'+rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    #### 结果集输出到csv文件 ####   
    # result.to_csv("D:\\trade_datas.csv", encoding="gbk", index=False)
    print(result)

    #### 登出系统 ####
    bs.logout()

def download_history_k_data_sample():
    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    #### 获取沪深A股历史K线数据 ####
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    rs = bs.query_history_k_data_plus("sh.600000",
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
        start_date='2024-07-01', end_date='2025-07-30',
        frequency="d", adjustflag="3")
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    #### 结果集输出到csv文件 ####   
    # result.to_csv("D:\\history_A_stock_k_data.csv", index=False)
    print(result)

    #### 登出系统 ####
    bs.logout()

if __name__ == '__main__':
    # 获取指定日期全部股票的日K线数据
    download_data_sample("2019-02-25")
    # trade_date_sample()
    # download_history_k_data_sample()
'''
参数名称	参数描述	算法说明
date	交易所行情日期	
code	证券代码	
open	开盘价	
high	最高价	
low	最低价	
close	收盘价	
preclose	前收盘价	见表格下方详细说明
volume	成交量（累计 单位：股）	
amount	成交额（单位：人民币元）	
adjustflag	复权状态(1：后复权， 2：前复权，3：不复权）	
turn	换手率	[指定交易日的成交量(股)/指定交易日的股票的流通股总股数(股)]*100%
tradestatus	交易状态(1：正常交易 0：停牌）	
pctChg	涨跌幅（百分比）	日涨跌幅=[(指定交易日的收盘价-指定交易日前收盘价)/指定交易日前收盘价]*100%
peTTM	滚动市盈率	(指定交易日的股票收盘价/指定交易日的每股盈余TTM)=(指定交易日的股票收盘价*截至当日公司总股本)/归属母公司股东净利润TTM
pbMRQ	市净率	(指定交易日的股票收盘价/指定交易日的每股净资产)=总市值/(最近披露的归属母公司股东的权益-其他权益工具)
psTTM	滚动市销率	(指定交易日的股票收盘价/指定交易日的每股销售额)=(指定交易日的股票收盘价*截至当日公司总股本)/营业总收入TTM
pcfNcfTTM	滚动市现率	(指定交易日的股票收盘价/指定交易日的每股现金流TTM)=(指定交易日的股票收盘价*截至当日公司总股本)/现金以及现金等价物净增加额TTM
isST	是否ST股，1是，0否""
'''

'''
参数名称	参数描述	说明
date	交易所行情日期	格式：YYYY-MM-DD
code	证券代码	格式：sh.600000。sh：上海，sz：深圳
open	今开盘价格	精度：小数点后4位；单位：人民币元
high	最高价	精度：小数点后4位；单位：人民币元
low	最低价	精度：小数点后4位；单位：人民币元
close	今收盘价	精度：小数点后4位；单位：人民币元
preclose	昨日收盘价	精度：小数点后4位；单位：人民币元
volume	成交数量	单位：股
amount	成交金额	精度：小数点后4位；单位：人民币元
adjustflag	复权状态	不复权、前复权、后复权
turn	换手率	精度：小数点后6位；单位：%
tradestatus	交易状态	1：正常交易 0：停牌
pctChg	涨跌幅（百分比）	精度：小数点后6位
peTTM	滚动市盈率	精度：小数点后6位
psTTM	滚动市销率	精度：小数点后6位
pcfNcfTTM	滚动市现率	精度：小数点后6位
pbMRQ	市净率	精度：小数点后6位
isST	是否ST	1是，0否
'''