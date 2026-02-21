#!/usr/bin/env python3
"""
股票策略回测主程序
"""
import sys
import os
import time
import json
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加当前目录到路径，确保能找到stock模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock.database.factory import RepositoryFactory
from stock.config import settings
from stock.data_context.context import DailyContext
from stock.data_context.concept_context_v2 import ConceptContext
from stock.selector.funnel import (
    FunnelSelector, 
    InitialUniverseStep, 
    ConceptRankingStep, 
    LiquidityFilterStep, 
    FinalSelectionStep
)
from stock.backtest.bt_backtester import BTBacktester, KSPPandasData
from stock.backtest.ksp_strategy import KSPStrategy
import backtrader as bt

def run_backtest(args=None):
    """
    运行回测主函数
    args: argparse.Namespace or object with attributes
    """
    print("="*70)
    print("🚀 股票策略回测系统启动 (Funnel Mode)")
    print("="*70)

    # 1. 参数处理
    start_date = getattr(args, 'start', "2025-01-01")
    if not start_date: start_date = "2025-01-01"
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    # 如果args有end属性且不为空，则使用
    if hasattr(args, 'end') and args.end:
        end_date = args.end
        
    slots = getattr(args, 'slots', 9)
    cash = getattr(args, 'cash', 1000000.0)
    top_concepts = getattr(args, 'top_concepts', 3)
    top_stocks = getattr(args, 'top_stocks', 3)
    sell_rank = getattr(args, 'sell_rank', 300)
    tp = getattr(args, 'tp', 0.10)
    sl = getattr(args, 'sl', -0.02)
    period = getattr(args, 'period', 5) # ksp_period
    min_amount = 50000000 # 恢复至 5000 万

    print(f"⚙️  配置: 资金={cash:,.0f}, 仓位={slots}, 止盈={tp:.0%}, 止损={sl:.0%}, 卖出排名>{sell_rank}")
    print(f"📅 周期: {start_date} ~ {end_date}")

    repo = RepositoryFactory.get_clickhouse_repo()

    # 2. 计算加载数据的起始时间 (start_date - 120天)
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    load_start_dt = start_dt - timedelta(days=120)
    load_start_date = load_start_dt.strftime('%Y-%m-%d')
    
    # 诊断 ClickHouse 连接与数据量 (更鲁棒的实现)
    print(f"🔍 正在诊断 ClickHouse 数据源...")
    try:
        db_res = repo.query("SELECT database()")
        db_name = db_res.iloc[0,0] if not db_res.empty else "UNKNOWN"
        
        count_res = repo.query(f"SELECT count(*) FROM {settings.TABLE_DAILY}")
        row_count = count_res.iloc[0,0] if not count_res.empty else 0
        
        range_res = repo.query(f"SELECT min(date), max(date) FROM {settings.TABLE_DAILY}")
        min_date = range_res.iloc[0,0] if not range_res.empty else "N/A"
        max_date = range_res.iloc[0,1] if not range_res.empty else "N/A"
        
        print(f"   📂 当前数据库: {db_name}, 表: {settings.TABLE_DAILY}")
        print(f"   📊 存量数据: {row_count:,} 行, 范围: {min_date} ~ {max_date}")
        
        if row_count == 0:
            print("⚠️ 警告: 数据库中没有行情数据，请检查数据同步状态。")
            repo.close()
            return
    except Exception as e:
        print(f"⚠️ 诊断过程出错: {e}")

    # 3. 加载行情数据
    print(f"\n📊 正在请求查询 (from {load_start_date})...")
    start_t = time.time()
    
    sql = f"SELECT * FROM {settings.TABLE_DAILY} WHERE date >= '{load_start_date}' ORDER BY date ASC"
    daily_df = repo.query(sql)
    
    if daily_df.empty:
        print(f"❌ 错误: SQL 查询返回空。SQL: {sql}")
        repo.close()
        return

    # 去重
    daily_df = daily_df.drop_duplicates(subset=['date', 'code'], keep='last')
    print(f"🚀 数据加载完成: {time.time()-start_t:.1f}s, {len(daily_df):,} 行")

    print("\n📈 加载基准数据 (上证指数)...")
    benchmark_df = repo.query(f"""
        SELECT date, close FROM {settings.TABLE_BENCHMARK}
        WHERE code = 'sh.000001' AND date >= '{start_date}' AND date <= '{end_date}'
        ORDER BY date ASC
    """)
    if benchmark_df.empty:
        print("⚠️  警告: 未找到基准数据，将不显示基准对比")
        benchmark_df = None

    # 3. 初始化策略 (使用新的 Funnel 架构)
    print("\n🔧 初始化策略上下文 (Funnel Pipeline)...")
    daily_ctx = DailyContext(daily_df=daily_df)
    concept_ctx = ConceptContext(repo=repo)
    
    # 构建漏斗步骤
    steps = [
        InitialUniverseStep(),                                  # Step 0: 基础过滤
        ConceptRankingStep(top_n=top_concepts),                 # Step 1: 概念优选
        LiquidityFilterStep(min_amount=min_amount),             # Step 2: 流动性过滤
        # 移除波动率筛选
        FinalSelectionStep(top_n_per_concept=top_stocks)        # Step 3: 最终 KSP 精选
    ]
    
    strategy_obj = FunnelSelector(daily_ctx, concept_ctx, steps)

    # 4. 执行回测
    print("\n" + "="*70)
    print("🎯 开始执行回测")
    print("="*70)

    # 初始化核心策略模块 (策略模式)
    from stock.strategy.ksp_core import KSPCore
    core_strategy = KSPCore(
        selector_obj=strategy_obj,
        slots=slots,
        sell_rank=sell_rank,
        take_profit=tp,
        stop_loss=sl
    )

    # 使用 KSPStrategy 适配器
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.0005)

    # 获取日期范围
    all_dates = sorted(daily_df['date'].unique())
    full_idx = pd.to_datetime(all_dates)
    
    # 加载基准数据
    if benchmark_df is not None:
        b_df = benchmark_df.copy()
        if 'date' in b_df.columns:
            b_df['datetime'] = pd.to_datetime(b_df['date'])
            b_df = b_df.set_index('datetime').reindex(full_idx).ffill().bfill()
            bm_feed = bt.feeds.PandasData(dataname=b_df, name='_master_clock_', plot=False)
            cerebro.adddata(bm_feed)
    else:
        dummy_df = pd.DataFrame(index=full_idx, data={'close': 1.0})
        cerebro.adddata(bt.feeds.PandasData(dataname=dummy_df, name='_master_clock_', plot=False))

    # 预过滤候选股票
    candidate_codes = set()
    for d_str in all_dates:
        dt = datetime.strptime(d_str, '%Y-%m-%d')
        if dt >= datetime.strptime(start_date, '%Y-%m-%d'):
            try:
                selection = strategy_obj.select(dt)
                if selection:
                    candidate_codes.update(selection)
            except:
                pass
    
    print(f"Total unique candidates to load: {len(candidate_codes)}")

    # 加载个股数据
    df_all = daily_df[daily_df['code'].isin(candidate_codes)].copy()
    df_all['datetime'] = pd.to_datetime(df_all['date'])
    df_all = df_all.set_index(['code', 'datetime']).sort_index()
    
    for code in candidate_codes:
        try:
            if code not in df_all.index.get_level_values(0):
                continue
            code_df = df_all.loc[(code, slice(None)), :].reset_index(level=0, drop=True)
            if code_df.empty:
                continue
            
            aligned_df = code_df.reindex(full_idx)
            fill_cols = ['open', 'high', 'low', 'close', 'poc', 'ksp_sum_5d_rank', 'list_days']
            for col in fill_cols:
                if col in aligned_df.columns:
                    if col in ['open', 'high', 'low', 'close', 'poc']:
                        aligned_df.loc[aligned_df[col] <= 0.01, col] = np.nan
                    aligned_df[col] = aligned_df[col].ffill().bfill()
            
            aligned_df['volume'] = aligned_df['volume'].fillna(0)
            if 'list_days' not in aligned_df.columns:
                aligned_df['list_days'] = 0
            aligned_df['list_days'] = aligned_df['list_days'].fillna(0)
            
            if aligned_df['close'].isna().all():
                continue
            
            data = KSPPandasData(dataname=aligned_df, name=code, plot=False)
            cerebro.adddata(data)
        except:
            continue

    # 添加策略适配器
    cerebro.addstrategy(
        KSPStrategy,
        core_strategy=core_strategy,
        slots=slots,
        log_file='backtest_detailed_log.json'
    )

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print("Starting Backtrader execution...")
    results = cerebro.run(runonce=False)
    strat = results[0] if results else None

    print(f"\n✅ 回测执行完成")

    # 5. 结果分析与保存
    if strat is not None:
        # 打印摘要
        returns = strat.analyzers.returns.get_analysis()
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        
        print("\n" + "="*40 + "\n      PROFESSIONAL BACKTEST REPORT      \n" + "="*40)
        print(f"Final Value:     {strat.broker.getvalue():,.2f}")
        print(f"Total Return:    {(strat.broker.getvalue()/cash - 1)*100:.2f}%")
        print(f"Max Drawdown:    {drawdown.max.drawdown:.2f}%")
        s_val = sharpe.get('sharperatio', 0)
        print(f"Sharpe Ratio:    {s_val if s_val is not None else 0:.2f}")
        
        if 'total' in trades:
            total = trades.total.total
            print(f"Total Trades:    {total}")
            if total > 0:
                print(f"Win Rate:        {(trades.won.total/total)*100:.2f}%")
        print("="*40)
        
        # 分析详细日志
        analyze_logs()
    
    repo.close()

def analyze_logs(log_file='backtest_detailed_log.json'):
    """分析策略生成的详细日志"""
    print("\n" + "="*70)
    print("📝 详细交易日志分析")
    print("="*70)

    if not os.path.exists(log_file):
        print("⚠️  未找到日志文件")
        return

    try:
        with open(log_file, 'r') as f:
            log_data = json.load(f)

        print(f"总交易记录: {len(log_data.get('trade_records', []))}")
        
        # 生成CSV报表
        trades = log_data.get('trade_records', [])
        if trades:
            df_trades = pd.DataFrame(trades)
            df_trades.to_csv('backtest_trades.csv', index=False)
            print(f"✅ 交易明细已保存: backtest_trades.csv")
            
            # 简单统计
            buys = df_trades[df_trades['action']=='BUY']
            sells = df_trades[df_trades['action']=='SELL']
            print(f"   买入: {len(buys)}, 卖出: {len(sells)}")
            
        daily = log_data.get('daily_records', [])
        if daily:
            df_daily = pd.DataFrame(daily)
            df_daily.to_csv('backtest_daily.csv', index=False)
            print(f"✅ 每日净值已保存: backtest_daily.csv")

    except Exception as e:
        print(f"❌ 日志分析出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行股票策略回测")
    parser.add_argument("--start", type=str, default="2025-01-01", help="开始日期")
    parser.add_argument("--end", type=str, default=None, help="结束日期")
    parser.add_argument("--cash", type=float, default=1000000.0, help="初始资金")
    parser.add_argument("--slots", type=int, default=9, help="最大持仓数")
    parser.add_argument("--top-concepts", type=int, default=3, help="选中概念数")
    parser.add_argument("--top-stocks", type=int, default=3, help="每个概念选股数")
    parser.add_argument("--sell-rank", type=int, default=400, help="卖出排名阈值")
    parser.add_argument("--tp", type=float, default=0.10, help="止盈比例")
    parser.add_argument("--sl", type=float, default=-0.02, help="止损比例")
    parser.add_argument("--period", type=int, default=5, help="KSP周期")
    
    args = parser.parse_args()
    run_backtest(args)
