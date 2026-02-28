#!/usr/bin/env python3
"""
股票策略回测主程序 - Fixed Parameter Logic
"""
import sys
import os
import time
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import backtrader as bt

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock.database.factory import RepositoryFactory
from stock.config import settings
from stock.data_context.context import DailyContext
from stock.data_context.concept_context_v2 import ConceptContext
from stock.selector.funnel import (
    FunnelSelector, InitialUniverseStep, ConceptRankingStep, RangeConceptRankingStep,
    LiquidityFilterStep, FinalSelectionStep, KSPMomentumStep
)
from stock.backtest.bt_backtester import KSPPandasData
from stock.backtest.ksp_strategy import KSPStrategy
from stock.backtest.data_factory import BTDataFeedFactory
from stock.strategy.modular_core import ModularKSPCore
from stock.strategy.rules import (
    RankEntryRule, RangeRankEntryRule, VolatilityConvergenceRule, VolumeRatioEntryRule,
    MovingAverageBiasRule,
    StopLossRule, TakeProfitRule, RankExitRule, BottomRankExitRule
)
from stock.backtest.reporting.trade_reporter import generate_trading_report
from stock.utils.health_check import DataHealthMonitor

def setup_output_dirs():
    """创建必要的输出目录"""
    dirs = ['logs', 'output/reports', 'output/plots']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def generate_filenames(strategy_id):
    """生成规范化的文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{strategy_id}_{timestamp}"
    return {
        'log': f"logs/{base_name}.json",
        'report': f"output/reports/{base_name}.md",
        'plot': f"output/plots/{base_name}.png"
    }

def plot_equity(log_file, save_path, strategy_id):
    """生成净值曲线图"""
    if not os.path.exists(log_file):
        return
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    daily_records = data.get('daily_records', [])
    if not daily_records:
        return

    df = pd.DataFrame(daily_records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    initial_cash = df['total_value'].iloc[0]
    df['equity'] = df['total_value'] / initial_cash
    df['cum_max'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] - df['cum_max']) / df['cum_max']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    ax1.plot(df['date'], df['equity'], color='blue', linewidth=1.5, label='Strategy Equity')
    ax1.set_title(f'KSP Strategy Equity Curve - {strategy_id}', fontsize=14)
    ax1.set_ylabel('Net Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.fill_between(df['date'], df['drawdown'], 0, color='red', alpha=0.3, label='Drawdown')
    ax2.set_ylabel('Drawdown %')
    ax2.set_ylim(df['drawdown'].min() * 1.2, 0.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def run_backtest(args):
    setup_output_dirs()
    filenames = generate_filenames(args.strategy_id)
    
    # 0. 数据健康自检
    monitor = DataHealthMonitor()
    if not monitor.validate_or_raise():
        print("🛑 由于数据滞后或不完整，回测已中止。请先运行数据更新任务。")
        return

    print("="*70)
    print(f"🚀 KSP Strategy - ID: {args.strategy_id}")
    print(f"📅 周期: {args.start} -> {args.end}")
    print(f"⚙️  参数: 仓位={args.slots}, 止损={args.sl:.0%}, 止盈={args.tp:.1%}")
    print(f"📝 日志: {filenames['log']}")
    print("="*70)

    start_date = args.start
    end_date = args.end
    slots = args.slots
    cash = args.cash
    tp, sl = args.tp, args.sl
    sell_rank = args.sell_rank

    repo = RepositoryFactory.get_clickhouse_repo()
    load_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=120)).strftime('%Y-%m-%d')
    
    daily_df = repo.query(f"SELECT * FROM {settings.TABLE_DAILY} WHERE date >= '{load_start_date}' AND date <= '{end_date}'")
    if daily_df.empty:
        print("❌ 错误: 数据库中没有行情数据")
        return

    benchmark_df = repo.query(f"SELECT date, close FROM {settings.TABLE_BENCHMARK} WHERE code = 'sh.000001' AND date >= '{start_date}' AND date <= '{end_date}'")

    daily_ctx = DailyContext(daily_df=daily_df)
    concept_ctx = ConceptContext(repo=repo)
    steps = [
        InitialUniverseStep(ksp_period=args.ksp_period),
        RangeConceptRankingStep(
            start_rank=args.concept_min_rank, 
            end_rank=args.concept_max_rank, 
            top_n=3, # 仅选择次优区间中最好的 3 个概念
            ksp_period=args.ksp_period
        ),
        LiquidityFilterStep(min_amount=50000000),
        FinalSelectionStep() # 传递概念下全部成分股
    ]
    strategy_obj = FunnelSelector(daily_ctx, concept_ctx, steps)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.0005)
    # 修改 full_idx 为包含预热期的完整时间轴
    # 这样 Backtrader 的指标 (如 MA20) 在 start_date 当天就已经计算完成了
    all_dates = sorted(daily_df['date'].unique())
    full_idx = pd.to_datetime(all_dates)
    
    if not benchmark_df.empty:
        # 基准数据也需要对齐到 full_idx
        cerebro.adddata(BTDataFeedFactory.create_benchmark_feed(benchmark_df, full_idx))
    else:
        dummy_df = pd.DataFrame(index=full_idx, data={'close': 1.0})
        cerebro.adddata(bt.feeds.PandasData(dataname=dummy_df, name='_master_clock_', plot=False))

    # 配置买入准入规则：区间排名 + 波动收敛 + 量比突破 + 均线乖离
    entry_rules = [
        RangeRankEntryRule(
            rank_col=f'ksp_sum_{args.ksp_period}d_rank', 
            min_rank=args.entry_min_rank, 
            max_rank=args.entry_rank
        ),
        VolatilityConvergenceRule(threshold=args.max_amp),
        VolumeRatioEntryRule(threshold=args.min_vol_ratio, window=5),
        MovingAverageBiasRule(window=20, min_bias=args.min_bias, max_bias=args.max_bias)
    ]
    
    # 配置卖出退出规则：排名劣化 (跌出 sell_rank) + 尾部风控 (D9-D10)
    exit_rules = [
        RankExitRule(rank_col=f'ksp_sum_{args.ksp_period}d_rank', threshold=args.sell_rank),
        BottomRankExitRule(rank_col='ksp_sum_5d_rank', bottom_threshold=args.tail_threshold)
    ]
    
    core_strategy = ModularKSPCore(
        selector_obj=strategy_obj, 
        entry_rules=entry_rules, 
        exit_rules=exit_rules,
        slots=slots
    )

    candidate_codes = set()
    for d_str in all_dates:
        dt = datetime.strptime(d_str, '%Y-%m-%d')
        if dt >= datetime.strptime(start_date, '%Y-%m-%d'):
            try:
                # 实时检查 KSP 因子覆盖率，若该日期没有任何排名数据，则说明数据不完整，中止回测
                day_data = daily_df[daily_df['date'] == d_str]
                ksp_col = f'ksp_sum_{args.ksp_period}d'
                if ksp_col in day_data.columns:
                    valid_ksp = day_data[day_data[ksp_col] != 0]
                    if len(valid_ksp) == 0:
                        print(f"\n🛑 严重错误: 日期 {d_str} 的 KSP ({args.ksp_period}d) 因子覆盖率为 0！")
                        print(f"   可能原因: 因子计算任务未运行或失败。请先运行 'python3 scripts/refresh_factors.py'。")
                        return

                selection = strategy_obj.select(dt)
                if selection: candidate_codes.update(selection)
            except Exception as e:
                print(f"⚠️  日期 {d_str} 选股失败: {e}")
    
    print(f"🎯 待加载标的总数: {len(candidate_codes)}")
    if not candidate_codes:
        print("⚠️  没有选出任何候选股，请检查数据。")
        return

    df_all = daily_df[daily_df['code'].isin(candidate_codes)].copy()
    df_all['datetime'] = pd.to_datetime(df_all['date'])
    for code in candidate_codes:
        code_df = df_all[df_all['code'] == code]
        feed = BTDataFeedFactory.create_stock_feed(code_df, code, full_idx)
        if feed is not None: cerebro.adddata(feed)
    
    cerebro.addstrategy(
        KSPStrategy, 
        core_strategy=core_strategy, 
        slots=slots, 
        ksp_period=args.ksp_period, 
        start_date=args.start, # 传入正式开始日期
        log_file=filenames['log']
    )

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    print("🚀 开始回测...")
    results = cerebro.run(runonce=False)
    
    if results:
        strat = results[0]
        # 回测结束后生成报告和图表
        print("\n📊 正在生成回测报告...")
        
        # 1. 生成交易报告 (Markdown)
        try:
            with open(filenames['log'], 'r') as f:
                log_data = json.load(f)
            generate_trading_report(log_data, filenames['report'], cash)
            print(f"✅ 交易报告已保存: {filenames['report']}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ 生成交易报告失败: {e}")

        # 2. 生成净值曲线图
        try:
            plot_equity(filenames['log'], filenames['plot'], args.strategy_id)
            print(f"✅ 净值曲线已保存: {filenames['plot']}")
        except Exception as e:
            print(f"❌ 生成净值曲线失败: {e}")

        trades = strat.analyzers.trades.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        
        print("\n" + "="*40)
        print(f"💰 最终价值: {strat.broker.getvalue():,.2f}")
        print(f"📈 累计收益: {(strat.broker.getvalue()/cash - 1)*100:.2f}%")
        print(f"📉 最大回撤: {drawdown.max.drawdown:.2f}%")
        if 'total' in trades:
            print(f"📊 总交易笔数: {trades.total.total}")
            won = trades.won.total if 'won' in trades else 0
            total = trades.total.total
            print(f"🏆 胜率: {won/total*100:.2f}%" if total > 0 else "胜率: N/A")
        print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行股票策略回测")
    parser.add_argument("--strategy_id", type=str, default="KSP_V7_Pro", help="策略ID，用于命名输出文件")
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--cash", type=float, default=1000000.0)
    parser.add_argument("--tp", type=float, default=0.099)
    parser.add_argument("--sl", type=float, default=-0.02)
    parser.add_argument("--entry_min_rank", type=int, default=440, help="买入排名最小门槛 (避开D1)")
    parser.add_argument("--entry_rank", type=int, default=1300, help="买入排名最大门槛 (D2-D3范围)")
    parser.add_argument("--max_amp", type=float, default=0.03, help="准入最大振幅限制 (波动收敛)")
    parser.add_argument("--min_vol_ratio", type=float, default=1.5, help="买入最小量比门槛 (U型启动)")
    parser.add_argument("--max_bias", type=float, default=0.05, help="最大均线乖离率 (不追高)")
    parser.add_argument("--min_bias", type=float, default=-0.03, help="最小均线乖离率 (回踩支撑)")
    parser.add_argument("--sell_rank", type=int, default=1500, help="卖出排名劣化门槛")
    parser.add_argument("--tail_threshold", type=int, default=3500, help="尾部退出排名门槛 (D9-D10)")
    parser.add_argument("--concept_min_rank", type=int, default=20, help="概念筛选起始排名 (避开极度过热)")
    parser.add_argument("--concept_max_rank", type=int, default=100, help="概念筛选结束排名")
    parser.add_argument("--top_concepts", type=int, default=3)
    parser.add_argument("--top_stocks", type=int, default=2)
    parser.add_argument("--ksp_period", type=int, default=5, help="KSP 排名周期 (5, 7, 10, 14)")
    args = parser.parse_args()
    run_backtest(args)
