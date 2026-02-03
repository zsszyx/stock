import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sql_op import sql_config

class SqlOp:
    def __init__(self, db_path=sql_config.db_path):
        # Increase timeout to 30 seconds to handle concurrent access better
        self.engine = create_engine(db_path, connect_args={'timeout': 30})
        # Enable Write-Ahead Logging (WAL) mode for better concurrency
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL;"))
            conn.execute(text("PRAGMA synchronous=NORMAL;")) # Optional: improves write performance

    def save(self, df: pd.DataFrame, table_name: str, index: bool = False):
        """
        Replaces the entire table with the dataframe. Use with caution.
        """
        with self.engine.begin() as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=index)
        return df

    def batch_insert(self, df: pd.DataFrame, table_name: str, index: bool = False, chunksize: int = 1000):
        """
        Fast insert (append) using multi-value insert.
        """
        if df.empty:
            return
        with self.engine.begin() as conn:
            df.to_sql(table_name, conn, if_exists='append', index=index, method='multi', chunksize=chunksize)

    def upsert_df_to_db(self, df: pd.DataFrame, table_name: str, index: bool = False):
        """
        Upserts DataFrame to database using a temporary table strategy.
        Target table must exist or it will be created.
        """
        if df.empty:
            return

        temp_table_name = f"temp_{table_name}"
        try:
            with self.engine.connect() as conn:
                # Check if target table exists
                if not self.engine.dialect.has_table(conn, table_name):
                    df.to_sql(table_name, self.engine, if_exists='fail', index=index)
                    return

            # Write to temp table
            with self.engine.begin() as conn:
                df.to_sql(temp_table_name, conn, if_exists='replace', index=index)
                
                # Get columns
                temp_df_cols = pd.read_sql(f'SELECT * FROM "{temp_table_name}" LIMIT 0', conn).columns.tolist()
                cols_str = ', '.join(f'"{col}"' for col in temp_df_cols)

                # INSERT OR REPLACE
                insert_sql = f"""
                    INSERT OR REPLACE INTO "{table_name}" ({cols_str})
                    SELECT {cols_str} FROM "{temp_table_name}"
                """
                conn.execute(text(insert_sql))
                
                # Drop temp table
                conn.execute(text(f'DROP TABLE IF EXISTS "{temp_table_name}"'))
                
        except Exception as e:
            print(f"Error in upsert_df_to_db for table {table_name}: {e}")

    def get_max_date_for_codes(self, codes: list, table_name: str) -> dict:
        """
        Query max date for a list of codes.
        """
        if not codes:
            return {}

        try:
            # Determine column name for date based on table convention
            date_col = 'date'
            
            # Use a more efficient IN query
            placeholders = ', '.join([f':code_{i}' for i in range(len(codes))])
            query_str = f"""
                SELECT code, MAX({date_col}) as max_date
                FROM {table_name}
                WHERE code IN ({placeholders})
                GROUP BY code
            """
            params = {f'code_{i}': code for i, code in enumerate(codes)}
            
            with self.engine.connect() as connection:
                result = connection.execute(text(query_str), params)
                max_dates = {row[0]: row[1] for row in result}
            
            return max_dates
        except Exception as e:
            print(f"Error getting max dates: {e}")
            return {}

    def query(self, query_str: str, parse_dates=None) -> pd.DataFrame:
        """
        Execute a generic SQL query.
        """
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query_str, conn)
                
            if parse_dates:
                for col in parse_dates:
                    if col in df.columns:
                        if col == 'time':
                            df[col] = pd.to_datetime(df[col], format='%Y%m%d%H%M%S%f', errors='coerce')
                        else:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
            return df
        except Exception as e:
            print(f"Query error: {e}")
            return None

    def read_k_data_by_date_range(self, table_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Read K-line data for a specific date range.
        """
        query_str = f"SELECT * FROM {table_name} WHERE date >= '{start_date}' AND date <= '{end_date}'"
        return self.query(query_str, parse_dates=['date', 'time'])

    def read_concept_constituent(self):
        query_str = f"SELECT * FROM {sql_config.concept_constituent_ths_table_name}"
        return self.query(query_str)

    def close(self):
        if self.engine:
            self.engine.dispose()

if __name__ == '__main__':
    sqlop = SqlOp()
    print("SqlOp initialized.")
    sqlop.close()