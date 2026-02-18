import pandas as pd
from typing import List, Dict, Set, Optional
from stock.database.base import BaseRepository
from stock.config import settings

class ConceptContext:
    """
    股票概念上下文，建立股票与概念的双向索引。
    数据现在默认从 ClickHouse 读取。
    """
    def __init__(self, df: pd.DataFrame = None, repo: Optional[BaseRepository] = None):
        if df is None:
            if repo is None:
                # Default to ClickHouse now
                from stock.database.factory import RepositoryFactory
                repo = RepositoryFactory.get_clickhouse_repo()
            self._df = self._load_from_repo(repo)
        else:
            self._df = df
            
        self._stock_to_concepts: Dict[str, Set[str]] = {}
        self._concept_to_stocks: Dict[str, Set[str]] = {}
        self._build_indexes()

    def _load_from_repo(self, repo: BaseRepository) -> pd.DataFrame:
        query = f"SELECT code, concept FROM {settings.TABLE_CONCEPT_CONSTITUENT_THS}"
        df = repo.query(query)
        
        if not df.empty:
            # Add sh./sz. prefix to match Baostock format
            def add_prefix(code):
                s_code = str(code)
                if s_code.startswith('6'):
                    return 'sh.' + s_code
                else:
                    return 'sz.' + s_code
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
