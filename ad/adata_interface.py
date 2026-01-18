import adata as ad
import pandas as pd

class AdataInterface:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_concept_constituent(self, index_code: str) -> pd.DataFrame:
        """
        获取概念成分股
        """
        try:
            df = ad.stock.info.concept_constituent_ths(index_code=index_code)
            return df
        except Exception as e:
            print(f"Failed to get concept constituent for {index_code}: {e}")
            return pd.DataFrame()

    def get_concept_min_ths(self, index_code: str) -> pd.DataFrame:
        """
        获取概念板块分钟行情
        """
        try:
            df = ad.stock.market.get_market_concept_min_ths(index_code=index_code)
            return df
        except Exception as e:
            print(f"Failed to get concept min ths for {index_code}: {e}")
            return pd.DataFrame()

    def get_concept_ths(self) -> pd.DataFrame:
        """
        获取所有概念
        """
        try:
            df = ad.stock.info.all_concept_code_ths()
            return df
        except Exception as e:
            print(f"Failed to get concept ths: {e}")
            return pd.DataFrame()
        
if __name__ == "__main__":
    with AdataInterface() as api:
        df = api.get_concept_ths()
        print(df)
        df = api.get_concept_constituent(index_code='886041')
        print(df)
        df = api.get_concept_min_ths(index_code='886041')
        print(df)