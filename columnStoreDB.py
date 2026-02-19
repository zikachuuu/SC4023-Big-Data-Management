import csv
import json
import numpy as np
import os
import gc
import pandas as pd
import math
import logging

from constants import INPUT_FILE, CHUNK_SIZE, TOWN_MAP_DIGIT, MONTH_MAP_DIGIT

logger = logging.getLogger("column_store_db")

class ColumnStoreDB:
    def __init__(self):

        # Dictionary to map column names to their column indexes (eg. "month" -> 0, "town" -> 1, etc.)
        self.col_names: dict[str, int] = {}

        # Compression: For each column, we mapped the increasingly sorted unique values (strings or int) to integer codes starting from 0.
        # This saves space and allows for faster comparisons.
        # To access the column, use the column index from col_names
        self.val_code_mapper: list[dict[str | int, int]] = [] 

        # Main Column Store data structure: List of numpy arrays, one for each column
        # Likewise, to access a column, use the column index from col_names
        # The data in these columns are the integer codes after mapping from original values using val_code_mapper
        self.columns: list[np.ndarray] = []

        # Number of rows and columns in the dataset
        self.row_count: int = 0
        self.col_count: int = 0

        # Number of chunks in the dataset based on CHUNK_SIZE
        self.num_chunks: int = 0

        # Zone Maps: For selected columns, we will store a list of metavalues for each chunk of rows (based on CHUNK_SIZE)
        # Outer list: Index corresponds to column index
        # Middle list: Index corresponds to chunk index
        # Inner list: Stores the metadata for that chunk. The content depends on the column:
        #   column "month"          -> [earliest month, latest month] for each chunk
        #   column "year"           -> [earliest year, latest year] for each chunk
        #   column "town"           -> [towns that appeared in the chunk]
        #   column "floor_area"     -> [min floor area, max floor area] for each chunk
        #   Other columns           -> No zone map (empty list)
        # The metedata in the zone maps are stored in terms of the integer codes after mapping from original values using val_code_mapper
        self.zone_maps: list[list[list[int]]] = []
    
    def _log_database_state(self):
        # Log the entire database to a file for debugging
        # Helper function to convert numpy types to JSON-serializable Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return [convert_to_serializable(item) for item in obj.tolist()]
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            return obj
        
        database_state = {
            "Column Store Database State": {
                "Column Names": self.col_names,
                "Value Code Mappers": [
                    {
                        col: {str(val): convert_to_serializable(code) for val, code in mapper.items()}
                    }
                    for col, mapper in enumerate(self.val_code_mapper)
                ],
                "Columns": [
                    {
                        col: convert_to_serializable(data)
                    }
                    for col, data in enumerate(self.columns)
                ],
                "Row Count": self.row_count,
                "Number of Chunks": self.num_chunks,
                "Zone Maps": [
                    {
                        col: {
                            "Chunks": [
                                {
                                    chunk: convert_to_serializable(zone)
                                }
                                for chunk, zone in enumerate(zones)
                            ]
                        }
                    }
                    for col, zones in enumerate(self.zone_maps)
                ]
            }
        }

        database_file_path = os.path.join(os.path.dirname(__file__), 'Logs', 'database_state.json')
        with open(database_file_path, 'w') as f:
            json.dump(database_state, f, indent=2)


    def load_csv(self, filepath):
        # Load the raw data
        df_temp: pd.DataFrame   = pd.read_csv(filepath)
        
        # Split the "month" column into "month" and "year", and insert the "year" column right after "month"
        month_split = df_temp["month"].str.split('-', expand=True)
        df_temp["month"] = month_split[0]  # Keep month name (Jan, Feb, etc.)

        month_idx = df_temp.columns.get_loc("month") # Insert year column right after month
        df_temp.insert(month_idx + 1, "year", month_split[1].astype(int))  # Convert year to int
        
        self.row_count          = len(df_temp)
        self.col_count          = len(df_temp.columns)
        self.col_names          = {col_name: i for i, col_name in enumerate(df_temp.columns)}

        self.val_code_mapper    = [None] * self.col_count  # Placeholder for the value to code mapping for each column
        self.columns            = [None] * self.col_count  # Placeholder for the encoded columns
        self.zone_maps          = [None] * self.col_count  # Placeholder for the zone maps for each column

        # Process Column by Column
        for (col_name, col_idx) in self.col_names.items():
            
            unique_vals: np.ndarray = df_temp[col_name].unique()

            if col_name == "month":
                # Use Custom mapping for Month to preserve chronological order (Jan < Feb < ... < Dec)
                self.val_code_mapper[col_idx] = {key: val for key, val in MONTH_MAP_DIGIT.items()}
            else:
                # Standard sorting (Lexicographical for strings, Numeric for ints)
                sorted_unique_vals = np.sort(unique_vals)
                # Stores the Original Value -> Integer Code mapping for this column
                self.val_code_mapper[col_idx] = {val: idx for idx, val in enumerate(sorted_unique_vals)}

            # Map the original column to integers
            encoded_col: np.ndarray = df_temp[col_name].map(self.val_code_mapper[col_idx]).to_numpy(dtype=np.int32)
            self.columns[col_idx] = encoded_col

            # We calculate metadata for chunks of rows
            self.num_chunks = math.ceil(self.row_count / CHUNK_SIZE)

            # Temp list to hold zone map info for this column
            zone_maps_col = []

            # Determine if this column needs a zone map
            if col_name in ["month", "year", "floor_area", "resale_price"]:
                # Min/Max Zone Map
                for i in range(self.num_chunks):
                    start_idx   : int           = i * CHUNK_SIZE
                    end_idx     : int           = min((i + 1) * CHUNK_SIZE, self.row_count)
                    chunk       : np.ndarray    = encoded_col[start_idx:end_idx]
                    
                    # Because we sorted the mapper, the integer codes preserve order!
                    # min(code) corresponds to min(value), max(code) to max(value).
                    zone_maps_col.append([np.min(chunk), np.max(chunk)])

            elif col_name == "town":
                # Unique Set Zone Map
                for i in range(self.num_chunks):
                    start_idx   : int           = i * CHUNK_SIZE
                    end_idx     : int           = min((i + 1) * CHUNK_SIZE, self.row_count)
                    chunk       : np.ndarray    = encoded_col[start_idx:end_idx]
                    
                    # Store unique town codes appearing in this chunk
                    zone_maps_col.append(np.unique(chunk).tolist())
            
            else:
                # No zone map for other columns
                zone_maps_col = []

            self.zone_maps[col_idx] = zone_maps_col            

        # Clean up temp DataFrame to save memory
        del df_temp
        gc.collect()

        # Log the database state after loading
        self._log_database_state()

        logger.info(f"Loaded CSV with {self.row_count} rows and {self.col_count} columns into Column Store Database.")

