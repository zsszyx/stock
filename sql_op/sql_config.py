import os
# db_path = 'sqlite:///d:\\stock\\stock.db'
mintues5_table_name = "mintues5"
stock_list_table_name = "stock_list"
trade_date_table_name = "trade_date"
stock_states_table_name = "stock_states"
concept_ths_table_name= "concept_ths"
concept_constituent_ths_table_name = "concept_constituent_ths"
concept_min_ths_table_name = "concept_min_ths"

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = f"sqlite:///{os.path.join(_base_dir, 'stock.db')}"