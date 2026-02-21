import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime
from stock.data_context.context import DailyContext
from stock.data_context.concept_context_v2 import ConceptContext

class BaseSelector:
    """Base class for all selectors to reduce boilerplate."""
    def __init__(self, daily_context: DailyContext):
        self.context = daily_context
    def get_data(self, date: datetime, window_days: int = 1, codes: Optional[List[str]] = None) -> pd.DataFrame:
        return self.context.get_window(date, window_days=window_days, codes=candidate_codes)
    def select(self, date: datetime, candidate_codes: Optional[List[str]] = None, **kwargs) -> List[str]:
        raise NotImplementedError("Subclasses must implement select()")

class KSPScoreSelector(BaseSelector):
    """
    KSP 分数选择器：
    支持选择不同周期 (5d, 7d, 14d) 的累计分数作为依据。
    """
    def __init__(self, daily_context: DailyContext, ksp_period: int = 14):
        super().__init__(daily_context)
        self.ksp_period = ksp_period

    def get_scores(self, date: datetime, candidate_codes: Optional[List[str]] = None) -> pd.DataFrame:
        df = self.context.get_window(date, window_days=1, codes=candidate_codes)
        if df.empty: return pd.DataFrame()

        score_col = f'ksp_sum_{self.ksp_period}d'
        if score_col not in df.columns:
            return pd.DataFrame()

        # 基础过滤：上市满 6 个月
        if 'list_days' in df.columns:
            df = df[df['list_days'] >= 180]

        result = df[['code', 'ksp_score', score_col, 'pct_chg', 'poc', 'amount']].copy() # Added 'amount'
        result['score'] = result[score_col]
        return result.sort_values('score', ascending=False)

    def select(self, date: datetime, candidate_codes: Optional[List[str]] = None, top_n: int = 10) -> List[str]:
        scores_df = self.get_scores(date, candidate_codes)
        return scores_df.head(top_n)['code'].tolist() if not scores_df.empty else []

class ConceptSelector(BaseSelector):
    """
    概念板块选择器 (v2 修改版)：
    1. 不做任何筛选，计算所有概念成分股的KSP分数并求均值，选出Top概念。
    2. 在胜出概念中，过滤掉当日交易额不足5000万的股票形成集合b。
    3. 在集合b中选出每个概念下的头三只股票形成持仓。
    """
    def __init__(self, 
                 score_selector: KSPScoreSelector, 
                 concept_context: ConceptContext,
                 top_concepts_n: int = 3,
                 top_stocks_per_concept_n: int = 3,
                 min_amount_filter: float = 50000000.0): # 5000万
        super().__init__(score_selector.context)
        self.score_selector = score_selector
        self.concept_context = concept_context
        self.top_concepts_n = top_concepts_n
        self.top_stocks_per_concept_n = top_stocks_per_concept_n
        self.min_amount_filter = min_amount_filter

    def select(self, date: datetime, candidate_codes: Optional[List[str]] = None, **kwargs) -> List[str]:
        """
        基于板块共振的选股。
        """
        # Step 1: 不做筛选获取全量分数
        scores_df = self.score_selector.get_scores(date, candidate_codes=None)
        if scores_df.empty: return []
        
        score_map = dict(zip(scores_df['code'], scores_df['score']))
        # 用于后续过滤交易额
        amount_map = dict(zip(scores_df['code'], scores_df['amount'])) if 'amount' in scores_df.columns else {}
        
        # Step 1.1: 统计每个概念板块的得分 (基于全集)
        concept_stats = []
        for concept in self.concept_context.all_concepts:
            constituent_codes = self.concept_context.get_stocks(concept)
            # 只要有分数的成分股都参与计算均值，不过滤
            valid_scores = [score_map[code] for code in constituent_codes if code in score_map]
            
            if valid_scores:
                concept_stats.append({
                    'concept': concept,
                    'avg_score': sum(valid_scores) / len(valid_scores),
                    'constituents': constituent_codes
                })
        
        if not concept_stats: return []
        
        # Step 1.2: 选取 Top 概念 (胜出的概念)
        top_concepts = pd.DataFrame(concept_stats).sort_values('avg_score', ascending=False).head(self.top_concepts_n)
        
        selected_stocks = []
        
        for _, row in top_concepts.iterrows():
            constituents = row['constituents']
            
            # Step 2: 过滤掉当日交易额不足5000万的股票形成集合b
            filtered_constituents = []
            for code in constituents:
                if code not in score_map: continue # 没数据的不要
                
                # 检查交易额 (如果数据里没有 amount 列，则跳过此过滤或视情况处理，这里假设有)
                amt = amount_map.get(code, 0)
                if amt < self.min_amount_filter:
                    continue
                
                filtered_constituents.append({'code': code, 'score': score_map[code]})
            
            # Step 3: 在集合b中选出每个概念下的头三只股票
            if filtered_constituents:
                sorted_b = pd.DataFrame(filtered_constituents).sort_values('score', ascending=False)
                # 选前N只
                top_b = sorted_b.head(self.top_stocks_per_concept_n)
                selected_stocks.extend(top_b['code'].tolist())
        
        # 4. 去重并返回
        seen = set()
        return [x for x in selected_stocks if not (x in seen or seen.add(x))]
