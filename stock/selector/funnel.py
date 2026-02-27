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
    def __init__(self, ksp_period: int = 5):
        super().__init__("Initial Universe")
        self.ksp_period = ksp_period
        self.ksp_col = f'ksp_sum_{ksp_period}d'

    def process(self, ctx: SelectorContext) -> SelectorContext:
        daily_df = ctx.daily_ctx.get_window(ctx.date, window_days=1)
        if daily_df.empty:
            ctx.pool = pd.DataFrame()
            return ctx

        mask = pd.Series(True, index=daily_df.index)
        if 'is_listed_180' in daily_df.columns:
            mask &= (daily_df['is_listed_180'] == 1)
        elif 'list_days' in daily_df.columns:
            mask &= (daily_df['list_days'] >= 180)
        
        if self.ksp_col in daily_df.columns:
            mask &= daily_df[self.ksp_col].notna()
            
        initial_pool = daily_df[mask].copy()
        
        cols = ['code', 'ksp_score', 'poc', 'amount', 'close', 'list_days', 'is_listed_180',
                'ksp_sum_5d', 'ksp_sum_10d', 'ksp_sum_5d_rank', 'ksp_sum_10d_rank']
        existing_cols = [c for c in cols if c in initial_pool.columns]
        ctx.pool = initial_pool[existing_cols].copy()
        
        ctx.log(f"Step 0 (Init): Loaded {len(ctx.pool)} stocks (list_days>=180, has_ksp)")
        return ctx

class KSPMomentumStep(FunnelStep):
    """
    新增步骤: KSP 动能筛选
    筛选 5日 KSP 分数 > 10日 KSP 分数 AND 5日 KSP 排名 < 10日 KSP 排名 (即排名提升)
    """
    def __init__(self):
        super().__init__("KSP Momentum")

    def process(self, ctx: SelectorContext) -> SelectorContext:
        if ctx.pool.empty: return ctx
        
        required = ['ksp_sum_5d', 'ksp_sum_10d', 'ksp_sum_5d_rank', 'ksp_sum_10d_rank']
        missing = [c for c in required if c not in ctx.pool.columns]
        if missing:
            ctx.log(f"Step KSP Momentum Warning: Missing columns {missing}")
            return ctx
            
        mask = (ctx.pool['ksp_sum_5d'] > ctx.pool['ksp_sum_10d']) & \
               (ctx.pool['ksp_sum_5d_rank'] < ctx.pool['ksp_sum_10d_rank'])
        
        ctx.pool = ctx.pool[mask].copy()
        return ctx

class ConceptRankingStep(FunnelStep):
    """
    步骤 1: 概念板块优选 (Concept Ranking)
    """
    def __init__(self, top_n: int = 3, ksp_period: int = 5):
        super().__init__("Concept Ranking")
        self.top_n = top_n
        self.ksp_period = ksp_period
        self.ksp_col = f'ksp_sum_{ksp_period}d'

    def process(self, ctx: SelectorContext) -> SelectorContext:
        if ctx.pool.empty: return ctx

        if self.ksp_col not in ctx.pool.columns:
            return ctx
            
        score_map = dict(zip(ctx.pool['code'], ctx.pool[self.ksp_col]))
        concept_stats = []
        
        for concept in ctx.concept_ctx.all_concepts:
            constituents = ctx.concept_ctx.get_stocks(concept)
            valid_scores = [score_map[c] for c in constituents if c in score_map]
            if valid_scores:
                concept_stats.append({
                    'concept': concept,
                    'avg_score': sum(valid_scores) / len(valid_scores)
                })
        
        if not concept_stats:
            ctx.pool = pd.DataFrame()
            return ctx
            
        top_concepts_df = pd.DataFrame(concept_stats).sort_values('avg_score', ascending=False).head(self.top_n)
        ctx.top_concepts = top_concepts_df['concept'].tolist()
        
        target_codes = set()
        for c in ctx.top_concepts:
            target_codes.update(ctx.concept_ctx.get_stocks(c))
            
        ctx.pool = ctx.pool[ctx.pool['code'].isin(target_codes)].copy()
        ctx.log(f"Step 1 (Active Concept): Selected top {self.top_n} concepts: {ctx.top_concepts}")
        return ctx

class RangeConceptRankingStep(FunnelStep):
    """
    步骤 1 (次优逻辑): 概念板块区间优选
    - 在排名区间 [start_rank, end_rank] 之间，选出均分最高的 top_n 个概念
    """
    def __init__(self, start_rank: int = 20, end_rank: int = 100, top_n: int = 3, ksp_period: int = 5):
        super().__init__("Range Concept Ranking")
        self.start_rank = start_rank
        self.end_rank = end_rank
        self.top_n = top_n
        self.ksp_period = ksp_period
        self.ksp_col = f'ksp_sum_{ksp_period}d'

    def process(self, ctx: SelectorContext) -> SelectorContext:
        if ctx.pool.empty: return ctx

        if self.ksp_col not in ctx.pool.columns:
            return ctx
            
        score_map = dict(zip(ctx.pool['code'], ctx.pool[self.ksp_col]))
        concept_stats = []
        
        for concept in ctx.concept_ctx.all_concepts:
            constituents = ctx.concept_ctx.get_stocks(concept)
            valid_scores = [score_map[c] for c in constituents if c in score_map]
            if valid_scores:
                concept_stats.append({
                    'concept': concept,
                    'avg_score': sum(valid_scores) / len(valid_scores)
                })
        
        if not concept_stats:
            ctx.pool = pd.DataFrame()
            return ctx
            
        df = pd.DataFrame(concept_stats).sort_values('avg_score', ascending=False)
        # 在区间 [start_rank, end_rank] 内取 top_n 名
        selected_df = df.iloc[self.start_rank-1 : self.end_rank].head(self.top_n)
        ctx.top_concepts = selected_df['concept'].tolist()
        
        target_codes = set()
        for c in ctx.top_concepts:
            target_codes.update(ctx.concept_ctx.get_stocks(c))
            
        ctx.pool = ctx.pool[ctx.pool['code'].isin(target_codes)].copy()
        ctx.log(f"Step 1 (Range Concept): Top {self.top_n} concepts from Rank {self.start_rank}-{self.end_rank}: {ctx.top_concepts}")
        return ctx

class LiquidityFilterStep(FunnelStep):
    """
    步骤 2: 流动性过滤
    """
    def __init__(self, min_amount: float = 50000000):
        super().__init__("Liquidity Filter")
        self.min_amount = min_amount

    def process(self, ctx: SelectorContext) -> SelectorContext:
        if ctx.pool.empty: return ctx
        if 'amount' not in ctx.pool.columns: return ctx
        ctx.pool = ctx.pool[ctx.pool['amount'] >= self.min_amount].copy()
        return ctx

class FinalSelectionStep(FunnelStep):
    """
    步骤 4: 全量成分股模式
    - 不再进行 Top N 过滤，而是直接让选定概念下的所有股票进入池子
    - 这为策略层的个股次优准入提供了更大的样本空间
    """
    def __init__(self):
        super().__init__("Final Selection")

    def process(self, ctx: SelectorContext) -> SelectorContext:
        if ctx.pool.empty: return ctx
        
        # 直接保留所有属于 top_concepts 且存活至今的股票
        ctx.log(f"Step 4 (Final): Passing all constituents of {len(ctx.top_concepts)} concepts to strategy core.")
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
        ctx = SelectorContext(self.daily_ctx, self.concept_ctx, date)
        for step in self.steps:
            ctx = step.process(ctx)
            if ctx.pool.empty:
                break
        return ctx.pool['code'].tolist() if not ctx.pool.empty else []
