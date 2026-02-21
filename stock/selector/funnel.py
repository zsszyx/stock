from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
from stock.data_context.context import DailyContext
from stock.data_context.concept_context_v2 import ConceptContext

class SelectorContext:
    """
    上下文对象，用于在漏斗的各个步骤间传递数据。
    """
    def __init__(self, daily_ctx: DailyContext, concept_ctx: ConceptContext, date: datetime):
        self.daily_ctx = daily_ctx
        self.concept_ctx = concept_ctx
        self.date = date
        self.pool: pd.DataFrame = pd.DataFrame()  # 当前候选股票池 (必须包含 'code' 列)
        self.top_concepts: List[str] = []         # 选出的热门概念
        self.logs: List[str] = []                 # 执行日志

    def log(self, message: str):
        self.logs.append(f"[{self.date.strftime('%Y-%m-%d')}] {message}")

class FunnelStep(ABC):
    """
    漏斗步骤基类
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, ctx: SelectorContext) -> SelectorContext:
        """
        执行过滤逻辑，修改 ctx.pool 或 ctx.top_concepts
        """
        pass

class InitialUniverseStep(FunnelStep):
    """
    步骤 0: 初始化股票池 (上市满180天)
    """
    def __init__(self):
        super().__init__("Initial Universe")

    def process(self, ctx: SelectorContext) -> SelectorContext:
        # 获取当日所有数据
        daily_df = ctx.daily_ctx.get_window(ctx.date, window_days=1)
        if daily_df.empty:
            ctx.pool = pd.DataFrame()
            return ctx

        # 过滤上市不满 180 天的
        # 注意: ksp_sum_5d_rank 等可能为 NaN
        mask = pd.Series(True, index=daily_df.index)
        if 'list_days' in daily_df.columns:
            mask &= (daily_df['list_days'] >= 180)
        
        # 必须有 ksp_sum_5d 分数
        if 'ksp_sum_5d' in daily_df.columns:
            mask &= daily_df['ksp_sum_5d'].notna()
            
        initial_pool = daily_df[mask].copy()
        
        # 保留必要的列
        cols = ['code', 'ksp_score', 'ksp_sum_5d', 'poc', 'amount', 'close', 'list_days']
        existing_cols = [c for c in cols if c in initial_pool.columns]
        ctx.pool = initial_pool[existing_cols].copy()
        
        ctx.log(f"Step 0 (Init): Loaded {len(ctx.pool)} stocks (list_days>=180, has_ksp)")
        return ctx

class ConceptRankingStep(FunnelStep):
    """
    步骤 1: 概念板块优选 (Concept Ranking)
    - 使用当前候选池 (ctx.pool) 中的活跃股票计算概念板块 KSP 均分
    - 选出 Top N 概念
    - 将当前的候选股票池限制在这些热门概念中
    """
    def __init__(self, top_n: int = 3):
        super().__init__("Concept Ranking")
        self.top_n = top_n

    def process(self, ctx: SelectorContext) -> SelectorContext:
        if ctx.pool.empty: return ctx

        # 准备分数映射 (仅基于当前池子里的活跃标的)
        if 'ksp_sum_5d' not in ctx.pool.columns:
            ctx.log("Step 1 Warning: 'ksp_sum_5d' column missing from pool.")
            return ctx
            
        score_map = dict(zip(ctx.pool['code'], ctx.pool['ksp_sum_5d']))
        
        concept_stats = []
        all_concepts = ctx.concept_ctx.all_concepts
        
        for concept in all_concepts:
            # 获取该概念下的成分股
            constituents = ctx.concept_ctx.get_stocks(concept)
            
            # 只计算在当前活跃池子里的股票分数
            valid_scores = [score_map[c] for c in constituents if c in score_map]
            
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                concept_stats.append({
                    'concept': concept,
                    'avg_score': avg_score,
                    'count': len(valid_scores)
                })
        
        if not concept_stats:
            ctx.log("Step 1: No valid concept scores calculated from active pool.")
            ctx.pool = pd.DataFrame()
            return ctx
            
        # 2. 选出 Top N 概念
        top_concepts_df = pd.DataFrame(concept_stats).sort_values('avg_score', ascending=False).head(self.top_n)
        ctx.top_concepts = top_concepts_df['concept'].tolist()
        
        # 3. 过滤 ctx.pool
        target_codes = set()
        for c in ctx.top_concepts:
            target_codes.update(ctx.concept_ctx.get_stocks(c))
            
        original_count = len(ctx.pool)
        ctx.pool = ctx.pool[ctx.pool['code'].isin(target_codes)].copy()
        
        ctx.log(f"Step 1 (Active Concept): Selected top {self.top_n} concepts from active pool: {ctx.top_concepts}")
        return ctx

class LiquidityFilterStep(FunnelStep):
    """
    步骤 2: 流动性过滤 (Liquidity Filter)
    """
    def __init__(self, min_amount: float = 50000000):
        super().__init__("Liquidity Filter")
        self.min_amount = min_amount

    def process(self, ctx: SelectorContext) -> SelectorContext:
        if ctx.pool.empty: return ctx
        
        original_count = len(ctx.pool)
        if 'amount' not in ctx.pool.columns:
            ctx.log("Step 2 Warning: 'amount' column missing, skipping filter.")
            return ctx
            
        ctx.pool = ctx.pool[ctx.pool['amount'] >= self.min_amount].copy()
        
        ctx.log(f"Step 2 (Liquidity): Filtered amount < {self.min_amount/10000:.0f}w")
        ctx.log(f"                   Reduced pool from {original_count} -> {len(ctx.pool)} stocks")
        return ctx

class WyckoffVolatilityStep(FunnelStep):
    """
    步骤 3 (新增): 威科夫波动率过滤 (Wyckoff Volatility Filter)
    - 计算过去 30 日波动率 (Std/Mean)
    - 保留波动率最低的前 30%
    """
    def __init__(self, window: int = 30, percentile: float = 0.30):
        super().__init__("Wyckoff Volatility")
        self.window = window
        self.percentile = percentile

    def process(self, ctx: SelectorContext) -> SelectorContext:
        if ctx.pool.empty: return ctx
        
        # 1. 获取历史数据 (30天)
        target_codes = list(ctx.pool['code'].unique())
        
        # 优化：批量获取历史数据
        # get_window 返回的是 DataFrame，包含 date, code, close 等
        history_df = ctx.daily_ctx.get_window(ctx.date, window_days=self.window, codes=target_codes)
        
        if history_df.empty:
            ctx.log("Step 3 (Wyckoff): No history data found.")
            return ctx
            
        # 2. 计算波动率 (Coefficient of Variation)
        # CV = StdDev / Mean
        # 先按代码分组计算
        grouped = history_df.groupby('code')['close']
        
        # 检查是否有足够的历史数据 (至少窗口的一半)
        min_history_required = self.window * 0.5
        
        stats = grouped.agg(['std', 'mean', 'count'])
        # 只保留数据足够的股票
        valid_stats = stats[stats['count'] >= min_history_required].copy()
        
        if valid_stats.empty:
            ctx.log(f"Step 3 (Wyckoff): WARNING - Not enough history for any stock (req={min_history_required}d). Skipping filter.")
            # 降级策略：如果完全没有足够的历史数据，则跳过此步骤，保留原股票池
            # 这样至少可以让回测跑起来 (虽然不仅失去了Wyckoff过滤)
            return ctx
            
        valid_stats['cv'] = valid_stats['std'] / valid_stats['mean']
        
        # 3. 筛选低波动
        # 找到分位阈值 (例如 30% 分位点)
        threshold = valid_stats['cv'].quantile(self.percentile)
        low_vol_codes = valid_stats[valid_stats['cv'] <= threshold].index.tolist()
        
        original_count = len(ctx.pool)
        ctx.pool = ctx.pool[ctx.pool['code'].isin(low_vol_codes)].copy()
        
        ctx.log(f"Step 3 (Wyckoff): Volatility Threshold (Top {self.percentile:.0%}) = {threshold:.4f}")
        ctx.log(f"                  Reduced pool from {original_count} -> {len(ctx.pool)} stocks")
        return ctx

class FinalSelectionStep(FunnelStep):
    """
    步骤 4: 最终精选 (Final Pick)
    - 在每个 Top Concept 中，选出 KSP 分数最高的 N 只
    """
    def __init__(self, top_n_per_concept: int = 3):
        super().__init__("Final Selection")
        self.top_n_per_concept = top_n_per_concept

    def process(self, ctx: SelectorContext) -> SelectorContext:
        if ctx.pool.empty: return ctx
        
        final_codes = set()
        # 准备分数映射
        score_map = dict(zip(ctx.pool['code'], ctx.pool['ksp_sum_5d']))
        
        # 遍历每个胜出的概念
        for concept in ctx.top_concepts:
            # 获取该概念下的成分股
            constituents = ctx.concept_ctx.get_stocks(concept)
            
            # 找到既属于该概念，又在当前幸存池(low vol + liquid)中的股票
            candidates = [c for c in constituents if c in score_map]
            
            # 排序并取 Top N
            candidates.sort(key=lambda x: score_map[x], reverse=True)
            top_candidates = candidates[:self.top_n_per_concept]
            final_codes.update(top_candidates)
            
        original_count = len(ctx.pool)
        ctx.pool = ctx.pool[ctx.pool['code'].isin(final_codes)].copy()
        
        ctx.log(f"Step 4 (Final): Selected top {self.top_n_per_concept} per concept")
        ctx.log(f"                Final selection: {len(ctx.pool)} stocks")
        return ctx

class FunnelSelector:
    """
    漏斗选择器 (Orchestrator)
    """
    def __init__(self, daily_ctx: DailyContext, concept_ctx: ConceptContext, steps: List[FunnelStep]):
        self.daily_ctx = daily_ctx
        self.concept_ctx = concept_ctx
        self.steps = steps
        
    def select(self, date: datetime) -> List[str]:
        # 初始化上下文
        ctx = SelectorContext(self.daily_ctx, self.concept_ctx, date)
        
        # 按顺序执行步骤
        for step in self.steps:
            ctx = step.process(ctx)
            if ctx.pool.empty:
                # print(f"Pool empty after step {step.name}") # debug
                break
                
        return ctx.pool['code'].tolist() if not ctx.pool.empty else []
