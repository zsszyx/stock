import pandas as pd
from typing import List, Optional
from datetime import datetime
from selector.score_selector import KSPScoreSelector
from data_context.concept_context import ConceptContext

from selector.base_selector import BaseSelector

class ConceptSelector(BaseSelector):
    def __init__(self, score_selector: KSPScoreSelector, concept_context: ConceptContext):
        super().__init__(score_selector.context)
        self.score_selector = score_selector
        self.concept_context = concept_context

    def select(self, date: datetime, candidate_codes: Optional[List[str]] = None, **kwargs) -> List[str]:
        """
        Implementation of the standard select() interface.
        """
        return self.select_top_stocks(date)

    def select_top_stocks(self, date: datetime, top_concepts_n: int = 3, top_stocks_per_concept_n: int = 3) -> List[str]:
        """
        Select top N concepts based on average KSP score, 
        then select top M stocks within each concept.
        """
        # 1. Get all stock scores for the day
        scores_df = self.score_selector.get_scores(date)
        if scores_df.empty:
            return []

        # Create a mapping for quick lookup
        score_map = dict(zip(scores_df['code'], scores_df['score']))
        
        # 2. Calculate average score for each concept
        concept_scores = []
        for concept in self.concept_context.all_concepts:
            constituent_codes = self.concept_context.get_stocks(concept)
            # Filter to those that have scores today
            valid_scores = [score_map[code] for code in constituent_codes if code in score_map]
            
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                concept_scores.append({
                    'concept': concept,
                    'avg_score': avg_score,
                    'constituents': constituent_codes
                })
        
        if not concept_scores:
            return []
            
        concept_scores_df = pd.DataFrame(concept_scores)
        # Sort concepts by average score descending
        top_concepts = concept_scores_df.sort_values('avg_score', ascending=False).head(top_concepts_n)
        
        # 3. From top concepts, pick top stocks
        selected_stocks = []
        for _, row in top_concepts.iterrows():
            constituents = row['constituents']
            # Sort these constituents by their individual score
            const_scores = []
            for code in constituents:
                if code in score_map:
                    const_scores.append({
                        'code': code,
                        'score': score_map[code]
                    })
            
            if const_scores:
                const_scores_df = pd.DataFrame(const_scores)
                top_constituents = const_scores_df.sort_values('score', ascending=False).head(top_stocks_per_concept_n)
                selected_stocks.extend(top_constituents['code'].tolist())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_selected = []
        for stock in selected_stocks:
            if stock not in seen:
                unique_selected.append(stock)
                seen.add(stock)
                
        return unique_selected
