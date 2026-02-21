import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

class ConceptContext:
    """
    概念板块上下文 - 带内存缓存优化
    
    优化点：
    1. 类级缓存 - 所有实例共享同一份概念数据
    2. TTL缓存 - 1小时过期，自动刷新
    3. 延迟加载 - 首次访问时才加载数据
    """
    
    # 类级缓存
    _cache: Optional[Dict[str, List[str]]] = None
    _cache_timestamp: Optional[datetime] = None
    _cache_ttl_seconds: int = 3600  # 1小时过期
    
    def __init__(self, repo=None):
        self.repo = repo
        self._ensure_cache_loaded()
    
    def _ensure_cache_loaded(self):
        """确保缓存已加载（延迟加载 + TTL检查）"""
        cache_expired = (
            ConceptContext._cache is None or
            ConceptContext._cache_timestamp is None or
            (datetime.now() - ConceptContext._cache_timestamp).seconds > ConceptContext._cache_ttl_seconds
        )
        
        if cache_expired:
            if ConceptContext._cache is None:
                print("[ConceptContext] Initializing cache...")
            else:
                print(f"[ConceptContext] Cache expired, refreshing...")
            
            self._load_cache_from_db()
    
    @staticmethod
    def _normalize_stock_code(code: str) -> str:
        """
        将股票代码标准化为与日线表一致的格式
        
        输入: 000531, 600983
        输出: sz.000531, sh.600983
        """
        code = str(code).strip()
        
        # 如果已经有前缀，直接返回
        if code.startswith('sz.') or code.startswith('sh.'):
            return code
        
        # 根据代码规则添加前缀
        # 上海: 600, 601, 603, 605, 688(科创板)
        # 深圳: 000, 001, 002(中小板), 003, 300(创业板), 301
        if len(code) == 6:
            if code.startswith(('600', '601', '603', '605', '688', '689')):
                return f'sh.{code}'
            elif code.startswith(('000', '001', '002', '003', '300', '301')):
                return f'sz.{code}'
        
        # 无法识别的代码，默认深圳
        return f'sz.{code}'
    
    def _load_cache_from_db(self):
        """从数据库加载所有概念数据到缓存"""
        if self.repo is None:
            from stock.database.factory import RepositoryFactory
            self.repo = RepositoryFactory.get_clickhouse_repo()
            close_after = True
        else:
            close_after = False
        
        try:
            # 一次性加载所有概念成分股
            from stock.config import settings
            query = f"""
                SELECT concept, code 
                FROM {settings.TABLE_CONCEPT_CONSTITUENT_THS}
                ORDER BY concept, code
            """
            df = self.repo.query(query)
            
            if df.empty:
                print("[ConceptContext] Warning: No concept data found in database")
                ConceptContext._cache = {}
            else:
                # 构建概念->成分股列表的映射字典（标准化代码格式）
                cache = {}
                for concept in df['concept'].unique():
                    raw_codes = df[df['concept'] == concept]['code'].tolist()
                    # 转换代码格式
                    normalized_codes = [self._normalize_stock_code(c) for c in raw_codes]
                    cache[concept] = normalized_codes
                
                ConceptContext._cache = cache
                print(f"[ConceptContext] Cache loaded: {len(cache)} concepts, "
                      f"{sum(len(v) for v in cache.values())} total constituent records")
            
            ConceptContext._cache_timestamp = datetime.now()
            
        finally:
            if close_after:
                self.repo.close()
    
    @property
    def all_concepts(self) -> List[str]:
        """获取所有概念名称列表"""
        self._ensure_cache_loaded()
        return list(ConceptContext._cache.keys())
    
    def get_stocks(self, concept: str) -> List[str]:
        """获取指定概念的成分股列表"""
        self._ensure_cache_loaded()
        return ConceptContext._cache.get(concept, [])
    
    def get_concept_by_stock(self, code: str) -> List[str]:
        """反向查询：获取某只股票所属的所有概念"""
        self._ensure_cache_loaded()
        concepts = []
        for concept, stocks in ConceptContext._cache.items():
            if code in stocks:
                concepts.append(concept)
        return concepts
    
    def get_concept_stats(self) -> pd.DataFrame:
        """获取概念统计信息"""
        self._ensure_cache_loaded()
        stats = []
        for concept, stocks in ConceptContext._cache.items():
            stats.append({
                'concept': concept,
                'constituent_count': len(stocks),
                'sample_stocks': ','.join(stocks[:3]) + ('...' if len(stocks) > 3 else '')
            })
        return pd.DataFrame(stats).sort_values('constituent_count', ascending=False)
    
    @classmethod
    def invalidate_cache(cls):
        """手动使缓存失效（当概念数据更新后调用）"""
        cls._cache = None
        cls._cache_timestamp = None
        print("[ConceptContext] Cache invalidated")
    
    @classmethod
    def get_cache_info(cls) -> dict:
        """获取缓存状态信息"""
        if cls._cache is None:
            return {'status': 'not_loaded', 'concepts': 0, 'age_seconds': None}
        
        age = (datetime.now() - cls._cache_timestamp).seconds if cls._cache_timestamp else None
        return {
            'status': 'loaded',
            'concepts': len(cls._cache),
            'total_constituents': sum(len(v) for v in cls._cache.values()),
            'age_seconds': age,
            'expires_in_seconds': max(0, cls._cache_ttl_seconds - age) if age else None
        }
