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
            df = ad.stock_board_concept_cons_ths(symbol=index_code)
            return df
        except Exception as e:
            print(f"Failed to get concept constituent for {index_code}: {e}")
            return pd.DataFrame()

    def get_concept_min_ths(self, index_code: str) -> pd.DataFrame:
        """
        获取概念板块分钟行情
        """
        try:
            df = ad.stock_board_concept_hist_min_ths(symbol=index_code)
            return df
        except Exception as e:
            print(f"Failed to get concept min ths for {index_code}: {e}")
            return pd.DataFrame()

    def get_concept_ths(self) -> pd.DataFrame:
        """
        获取所有概念
        """
        try:
            df = ad.stock_board_concept_name_ths()
            return df
        except Exception as e:
            print(f"Failed to get concept ths: {e}")
            return pd.DataFrame()