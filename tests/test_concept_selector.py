import pytest
import pandas as pd
from unittest.mock import MagicMock
from datetime import datetime
from selector.concept_selector import ConceptSelector

def test_concept_selector_selection():
    # Mock KSPScoreSelector
    mock_score_selector = MagicMock()
    # Mock ConceptContext
    mock_concept_context = MagicMock()
    
    # Setup mock data
    # Stocks: S1, S2, S3 (Concept A), S4, S5, S6 (Concept B), S7, S8, S9 (Concept C)
    scores_df = pd.DataFrame({
        'code': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10'],
        'score': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    })
    mock_score_selector.get_scores.return_value = scores_df
    
    mock_concept_context.all_concepts = ['A', 'B', 'C', 'D']
    mock_concept_context.get_stocks.side_effect = lambda c: {
        'A': ['S1', 'S2', 'S3'],
        'B': ['S4', 'S5', 'S6'],
        'C': ['S7', 'S8', 'S9'],
        'D': ['S10']
    }[c]
    
    selector = ConceptSelector(mock_score_selector, mock_concept_context)
    date = datetime(2025, 1, 1)
    
    selected_stocks = selector.select_top_stocks(date)
    
    # Concept A mean: (10+9+8)/3 = 9
    # Concept B mean: (7+6+5)/3 = 6
    # Concept C mean: (4+3+2)/3 = 3
    # Concept D mean: 1
    # Top 3 concepts: A, B, C
    # Top 3 stocks in A: S1, S2, S3
    # Top 3 stocks in B: S4, S5, S6
    # Top 3 stocks in C: S7, S8, S9
    
    assert len(selected_stocks) == 9
    assert set(selected_stocks) == {'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'}
