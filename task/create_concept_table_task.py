
import sys
import os
import pandas as pd
from tqdm import tqdm

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ad.adata_interface import AdataInterface
from sql_op.op import SqlOp
from sql_op import sql_config

class CreateConceptTableTask:
    def __init__(self):
        self.sql_op = SqlOp()

    def run(self):
        """
        Creates a table of all stocks and their corresponding concepts.
        """
        all_stocks_concepts = []
        with AdataInterface() as api:
            # 1. Get all concepts
            concepts_df = api.get_concept_ths()
            if concepts_df.empty:
                print("No concepts found.")
                return

            # 2. Iterate through each concept and get its constituents
            for _, concept_row in tqdm(concepts_df.iterrows(), total=concepts_df.shape[0]):
                concept_name = concept_row['name']
                concept_code = concept_row['index_code']
                
                constituents_df = api.get_concept_constituent(index_code=concept_code)
                
                if not constituents_df.empty:
                    for _, stock_row in constituents_df.iterrows():
                        all_stocks_concepts.append({
                            'name': stock_row['short_name'],
                            'code': stock_row['stock_code'],
                            'concept': concept_name,
                            'concept_code': concept_code
                        })
                else:
                    print(f"Warning: No constituents found for concept: {concept_name} ({concept_code})")

        # 3. Create a DataFrame and save to DB
        if all_stocks_concepts:
            result_df = pd.DataFrame(all_stocks_concepts)
            result_df.set_index(['code', 'concept_code'], inplace=True)
            self.sql_op.save(result_df, sql_config.concept_constituent_ths_table_name, index=True)
            print("Successfully saved concept constituents to the database.")
        else:
            print("No constituent stocks found for any concept.")

if __name__ == "__main__":
    task = CreateConceptTableTask()
    task.run()