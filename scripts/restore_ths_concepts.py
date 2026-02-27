import pandas as pd
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

sys.path.insert(0, os.getcwd())
from stock.data_fetch.data_provider.adata_provider import AdataInterface
from stock.database.factory import RepositoryFactory
from stock.config import settings

def fetch_and_store_concept(interface, concept_name, index_code):
    """抓取单个概念的成分股"""
    try:
        df = interface.get_concept_constituent(index_code=index_code)
        if df is not None and not df.empty:
            # 找到包含 'code' 字样的列，adata 通常返回 'stock_code'
            code_col = [c for c in df.columns if 'code' in c.lower()]
            if not code_col:
                return pd.DataFrame()
            
            res = pd.DataFrame()
            res['concept'] = [concept_name] * len(df)
            res['code'] = df[code_col[0]].astype(str)
            return res
    except Exception as e:
        print(f"❌ 抓取概念成分 [{concept_name}] 发生严重异常: {e}")
        # 在 Fail-Fast 模式下，单只失败仅记录，但如果全量失败需关注
    return pd.DataFrame()

def restore_ths_concepts():
    print("🚀 启动同花顺 (THS) 概念表严格恢复流程...")
    
    with AdataInterface() as interface:
        print("📥 正在拉取全量概念列表...")
        concept_list_df = interface.get_concept_ths()
        
        if concept_list_df.empty:
            print("❌ 无法获取概念列表。")
            return
            
        print(f"📊 发现 {len(concept_list_df)} 个同花顺概念。开始抓取成分股映射...")
        
        all_mappings = []
        # 使用线程池，adata 对频率有一定限制
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_concept = {
                executor.submit(fetch_and_store_concept, interface, row['name'], row['index_code']): row['name']
                for _, row in concept_list_df.iterrows()
            }
            
            for future in tqdm(as_completed(future_to_concept), total=len(future_to_concept), desc="抓取概念成分"):
                df_mapping = future.result()
                if not df_mapping.empty:
                    all_mappings.append(df_mapping)
                # 稍微增加一点随机延迟
                time.sleep(0.1)

        if all_mappings:
            final_df = pd.concat(all_mappings, ignore_index=True)
            # 必须严格对齐 ClickHouse 的 (code, concept) 顺序
            final_df = final_df[['code', 'concept']]
            
            # 存入数据库
            repo = RepositoryFactory.get_clickhouse_repo()
            repo.execute(f"TRUNCATE TABLE {settings.TABLE_CONCEPT_CONSTITUENT_THS}")
            repo.insert_df(final_df, settings.TABLE_CONCEPT_CONSTITUENT_THS)
            
            print(f"✅ 同花顺概念表恢复完成！共记录 {len(final_df)} 条映射关系。")
            repo.close()
        else:
            print("❌ 未抓取到有效的成分股映射。")

if __name__ == "__main__":
    restore_ths_concepts()
