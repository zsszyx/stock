import pandas as pd
from datetime import datetime
from typing import Dict, Any
from stock.database.factory import RepositoryFactory
from stock.config import settings

class DataHealthMonitor:
    """
    数据健康监测器：
    - 检查行情数据滞后
    - 检查因子计算完整性
    - 检查数据断层 (Gaps)
    """
    def __init__(self, repo=None):
        self.repo = repo or RepositoryFactory.get_clickhouse_repo()

    def get_last_trading_date(self) -> str:
        """获取全市场的最后一个交易日 (基于基准指数)"""
        try:
            query = f"SELECT max(date) as max_date FROM {settings.TABLE_BENCHMARK}"
            res = self.repo.query(query)
            if not res.empty and pd.notna(res.iloc[0]['max_date']):
                return str(res.iloc[0]['max_date'])
            return datetime.now().strftime('%Y-%m-%d')
        except:
            return datetime.now().strftime('%Y-%m-%d')

    def check_health(self) -> Dict[str, Any]:
        """执行全面健康检查"""
        last_market_date = self.get_last_trading_date()
        
        # 1. 检查行情滞后
        max_daily_query = f"SELECT max(date) as max_date FROM {settings.TABLE_DAILY}"
        max_daily_res = self.repo.query(max_daily_query)
        db_max_date = str(max_daily_res.iloc[0]['max_date']) if not max_daily_res.empty and pd.notna(max_daily_res.iloc[0]['max_date']) else "1970-01-01"
        
        # 2. 检查因子完整性 (检查最新日期的 KSP 分数)
        factor_query = f"""
        SELECT 
            count(*) as total_stocks,
            countIf(ksp_sum_5d != 0) as count_5d,
            countIf(ksp_sum_10d != 0) as count_10d,
            countIf(poc > 0) as poc_count
        FROM {settings.TABLE_DAILY}
        WHERE date = '{db_max_date}'
        """
        f_res = self.repo.query(factor_query)
        
        # 3. 检查分钟线滞后
        max_min5_query = f"SELECT max(date) as max_date FROM {settings.TABLE_MIN5}"
        m5_res = self.repo.query(max_min5_query)
        db_min5_date = str(m5_res.iloc[0]['max_date']) if not m5_res.empty and pd.notna(m5_res.iloc[0]['max_date']) else "1970-01-01"

        status = {
            "market_last_date": last_market_date,
            "daily_max_date": db_max_date,
            "min5_max_date": db_min5_date,
            "is_daily_lagging": db_max_date < last_market_date,
            "is_min5_lagging": db_min5_date < last_market_date,
            "factor_integrity_5d": 0.0,
            "factor_integrity_10d": 0.0,
            "poc_integrity": 0.0,
            "is_healthy": True,
            "warnings": []
        }

        if not f_res.empty and f_res.iloc[0]['total_stocks'] > 0:
            total = f_res.iloc[0]['total_stocks']
            status["factor_integrity_5d"] = float(f_res.iloc[0]['count_5d'] / total)
            status["factor_integrity_10d"] = float(f_res.iloc[0]['count_10d'] / total)
            status["poc_integrity"] = float(f_res.iloc[0]['poc_count'] / total)

        # 健康判定
        if status["is_daily_lagging"]:
            status["warnings"].append(f"日线数据滞后! 市场最后日期: {last_market_date}, 数据库最后日期: {db_max_date}")
            status["is_healthy"] = False
        
        if status["factor_integrity_5d"] < 0.9:
            status["warnings"].append(f"5日因子不完整! 覆盖率: {status['factor_integrity_5d']:.1%}")
            status["is_healthy"] = False

        if status["factor_integrity_10d"] < 0.9:
            status["warnings"].append(f"10日因子不完整! 覆盖率: {status['factor_integrity_10d']:.1%}")
            status["is_healthy"] = False

        if status["poc_integrity"] < 0.9:
            status["warnings"].append(f"POC 数据缺失! 最新日期覆盖率: {status['poc_integrity']:.1%}")
            status["is_healthy"] = False

        return status

    def validate_or_raise(self):
        """如果数据不健康，抛出异常或打印严重警告"""
        status = self.check_health()
        print("\n" + "="*50)
        print("🔍 数据健康自检报告")
        print(f"  - 市场最后交易日: {status['market_last_date']}")
        print(f"  - 数据库最后日线: {status['daily_max_date']} ({'滞后' if status['is_daily_lagging'] else '同步'})")
        print(f"  - 5日因子覆盖率: {status['factor_integrity_5d']:.1%}")
        print(f"  - 10日因子覆盖率: {status['factor_integrity_10d']:.1%}")
        print(f"  - POC覆盖率: {status['poc_integrity']:.1%}")
        
        if not status["is_healthy"]:
            print("\n❌ 数据完整性存在隐患！")
            for w in status["warnings"]:
                print(f"    ⚠️  {w}")
            print("="*50 + "\n")
            return False
        
        print("\n✅ 数据健康检查通过。")
        print("="*50 + "\n")
        return True

if __name__ == "__main__":
    monitor = DataHealthMonitor()
    monitor.validate_or_raise()
