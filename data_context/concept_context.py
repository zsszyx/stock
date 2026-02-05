import pandas as pd
import sqlite3
from typing import List, Dict, Set
from stock.sql_op import sql_config

class ConceptContext:
    """
    股票概念上下文，建立股票与概念的双向索引。
    """
    def __init__(self, df: pd.DataFrame = None):
        if df is None:
            self._df = self._load_from_db()
        else:
            self._df = df
            
        self._stock_to_concepts: Dict[str, Set[str]] = {}
        self._concept_to_stocks: Dict[str, Set[str]] = {}
        self._build_indexes()

    def _load_from_db(self) -> pd.DataFrame:
        db_file_path = sql_config.db_path.replace("sqlite:///", "")
        query = f"SELECT code, concept FROM {sql_config.concept_constituent_ths_table_name}"
        with sqlite3.connect(db_file_path) as conn:
            df = pd.read_sql_query(query, conn)
            # Add sh./sz. prefix to match Baostock format
            def add_prefix(code):
                if code.startswith('6'):
                    return 'sh.' + code
                else:
                    return 'sz.' + code
            df['code'] = df['code'].apply(add_prefix)
            return df

    def _build_indexes(self):
        for _, row in self._df.iterrows():
            stock = row['code']
            concept = row['concept']
            
            # 正排索引: Stock -> Concepts
            if stock not in self._stock_to_concepts:
                self._stock_to_concepts[stock] = set()
            self._stock_to_concepts[stock].add(concept)
            
            # 倒排索引: Concept -> Stocks
            if concept not in self._concept_to_stocks:
                self._concept_to_stocks[concept] = set()
            self._concept_to_stocks[concept].add(stock)

    def get_concepts(self, code: str) -> List[str]:
        """获取某只股票所属的所有概念"""
        return list(self._stock_to_concepts.get(code, []))

    def get_stocks(self, concept: str) -> List[str]:
        """获取属于某个概念的所有股票"""
        return list(self._concept_to_stocks.get(concept, []))

    @property
    def all_concepts(self) -> List[str]:
        return list(self._concept_to_stocks.keys())

    @property
    def all_stocks(self) -> List[str]:
        return list(self._stock_to_concepts.keys())

    def get_common_concepts(self, codes: List[str]) -> Dict[str, int]:
        """统计一组股票中出现频率最高的概念"""
        counts = {}
        for code in codes:
            concepts = self.get_concepts(code)
            for cp in concepts:
                counts[cp] = counts.get(cp, 0) + 1
        # 按频率降序排列
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
