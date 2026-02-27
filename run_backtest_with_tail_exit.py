#!/usr/bin/env python3
"""
股票策略回测 - 使用 D9-D10 退出逻辑
"""
import sys
import os
import argparse
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_backtest import setup_output_dirs, generate_filenames, run_backtest
from stock.strategy.rules import (
    RankEntryRule, MomentumEntryRule, 
    BottomRankExitRule, MomentumFlipRule, RankExitRule
)

# 覆盖 run_backtest 中的逻辑或者直接在此处实现一个简化版
# 为了保持模块化，我们稍微修改 run_backtest.py 的结构使其更易调用，或者直接在这里重写核心配置部分。

# 实际上，我可以修改 run_backtest.py 里的 exit_rules 配置逻辑，
# 或者直接在这里定义一个新的 run 逻辑。

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行带尾部退出的股票策略回测")
    parser.add_argument("--strategy_id", type=str, default="KSP_Tail_Exit", help="策略ID")
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--cash", type=float, default=1000000.0)
    parser.add_argument("--tp", type=float, default=0.099)
    parser.add_argument("--sl", type=float, default=-0.02)
    parser.add_argument("--entry_rank", type=int, default=250)
    parser.add_argument("--sell_rank", type=int, default=300)
    parser.add_argument("--tail_threshold", type=int, default=3500, help="尾部退出排名门槛 (D9-D10)")
    parser.add_argument("--top_concepts", type=int, default=3)
    parser.add_argument("--top_stocks", type=int, default=2)
    parser.add_argument("--ksp_period", type=int, default=5)
    args = parser.parse_args()

    # 这里的代码大部分会复用 run_backtest.py 的逻辑，
    # 但为了演示模块化规则注入，我们直接修改 exit_rules。
    
    # 注意：在生产中，我们会重构 run_backtest.py 使其支持传入 rules。
    # 这里我们演示如何组合规则：
    
    print(f"🛠️  配置退出规则: D9-D10 尾部退出 (排名 > {args.tail_threshold})")
    
    # 我们可以通过 monkey-patch 或者直接调用 run_backtest 里的组件
    # 下面是一个“注入”新规则的示例：
    
    # 重新定义 exit_rules 的生成逻辑并运行
    # (实际上更建议修改 run_backtest.py 使其更通用，但这里先完成任务)
    
    # 由于 run_backtest 里的 exit_rules 是硬编码的，我需要在这里重写 run_backtest 
    # 或者修改 run_backtest.py 使其接受规则。
    
    # 决定：修改 run_backtest.py 使其更模块化，允许外部指定规则。
    pass
