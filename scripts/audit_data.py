import pandas as pd
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stock.database.factory import RepositoryFactory
from stock.config import settings

def audit_data():
    repo = RepositoryFactory.get_clickhouse_repo()
    print("="*50)
    print("🔍 启动 ClickHouse 数据源审计...")
    print("="*50)
    
    # 1. 基础价格逻辑审计
    print("\n1. 正在检查价格一致性 (High >= Open/Close, Low <= Open/Close)...")
    query = f"SELECT count(*) as bad_rows FROM {settings.TABLE_DAILY} WHERE high < open OR high < close OR low > open OR low > close"
    bad_rows = repo.query(query).iloc[0]['bad_rows']
    if bad_rows > 0:
        print(f"❌ 警告: 发现 {bad_rows} 行数据存在价格逻辑矛盾！")
    else:
        print("✅ 价格一致性检查通过。")

    # 2. 关键列缺失审计
    print("\n2. 正在检查关键列缺失情况 (poc, amount)...")
    query = f"SELECT count(*) as total FROM {settings.TABLE_DAILY} WHERE poc IS NULL OR amount IS NULL"
    missing = repo.query(query).iloc[0]['total']
    if missing > 0:
        print(f"⚠️ 警告: 发现 {missing} 行数据的 POC 或成交额缺失。")
    else:
        print("✅ 关键列完整性检查通过。")

    # 3. 涨跌幅异常审计
    print("\n3. 正在检查异常涨跌幅 (单日 > 25%)...")
    query = f"SELECT date, code, open, close, (close-open)/open as chg FROM {settings.TABLE_DAILY} WHERE abs((close-open)/open) > 0.25 LIMIT 5"
    anomalies = repo.query(query)
    if not anomalies.empty:
        print(f"⚠️ 警告: 发现疑似异常的单日涨跌幅。")
        print(anomalies)
    else:
        print("✅ 未发现超常规涨跌幅数据。")

    # 4. 样本抽样审计
    code = 'sh.600000'
    print(f"\n4. 正在对样本股票 {code} 进行抽样验证...")
    sample = repo.query(f"SELECT date, code, open, high, low, close, amount, poc FROM {settings.TABLE_DAILY} WHERE code = '{code}' ORDER BY date DESC LIMIT 5")
    print(sample)

    print("\n" + "="*50)
    print("✅ 审计完成。")
    print("="*50)

if __name__ == "__main__":
    audit_data()
