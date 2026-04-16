"""
columnStoreDB.py contains the ColumnStoreDB class that will be used to implement the column store database.
"""

import csv
import json
import numpy as np
import os
import gc
import pandas as pd
import math
import logging
from BTrees.IOBTree import IOBTree

from constants import CHUNK_SIZE, LOG_DIR, DB_DIR
from utility import convert_month_str_to_code, convert_floor_area_to_code

logger = logging.getLogger("column_store_db")

class ColumnStoreDB:
    """
    A disk-based columnar database that stores each column as a memory-mapped
    binary file (np.memmap) with integer-encoded values.

    Refer to report for the design features of the database.
    """

    def __init__(self):
        """
        Initialise an empty ColumnStoreDB.

        List of attributes:
            col_names       : dict[str, int]              - column name to column index.
            val_code_mapper : list[dict[str|int, int]]    - per-column mapping from raw value to integer code.
            code_val_mapper : list[dict[int, str|int]]    - per-column mapping from integer code to raw value.
            columns         : list[np.memmap]             - per-column memory-mapped int32 array.
            row_count       : int                         - total number of rows loaded.
            col_count       : int                         - total number of columns.
            num_chunks      : int                         - number of CHUNK_SIZE chunks (for zone maps).
            zone_maps       : list[list[list[int]]]       - per-column, per-chunk zone map data.
            month_btree     : IOBTree                     - B-Tree index mapping month_code to (start_row, end_row).
        """
        self.col_names: dict[str, int] = {}
        self.val_code_mapper: list[dict[str | int, int]] = [] 
        self.code_val_mapper: list[dict[int, str | int]] = []  
        self.columns: list[np.memmap] = []
        self.row_count: int = 0
        self.col_count: int = 0
        self.num_chunks: int = 0
        self.zone_maps: list[list[list[int]]] = []
        self.month_btree = IOBTree() 
    
    def _log_database_state(self):
        """
        Save the current database metadata (column names, row count,
        chunk count, and B-Tree contents) to a JSON file for debugging purposes.
        """
        def convert_to_serializable(obj):
            """Recursively convert NumPy types and memmap arrays to native
            Python types so they can be serialised by json.dump."""
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.memmap): return [convert_to_serializable(item) for item in obj.tolist()]
            elif isinstance(obj, list): return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
            return obj
        
        database_state = {
            "Column Store Database State": {
                "Column Names": self.col_names,
                "Row Count": self.row_count,
                "Number of Chunks": self.num_chunks,
                "Month B-Tree": {k: v for k, v in self.month_btree.items()}
            }
        }

        database_file_path = os.path.join(LOG_DIR, 'database_state.json') if 'LOG_DIR' in globals() else os.path.join(DB_DIR, 'database_state.json')
        os.makedirs(os.path.dirname(database_file_path), exist_ok=True)
        with open(database_file_path, 'w') as f:
            json.dump(database_state, f, indent=2)

    def load_csv(self, filepath):
        """
        Process the CSV file into the columnar database.

        We divide this process into four passes:

        Pass 0 - Physical clustering:
        Reads the entire CSV with pandas, sorts it by month_code so that
        rows with the same month are contiguous on disk, then writes a
        temporary clustered CSV to DB_DIR.

        Pass 1 - Retrieving unique values and dictionary encoding setup:
        Scans each column's unique values and builds the val_code_mapper
        and code_val_mapper dictionaries. Special encoding is used for
        'month' and 'floor_area_sqm' to preserve sort order and precision.
        Refer to report for specific details.

        Pass 2 - Chunk-wise encoding, disk write and B-tree index construction:
        Reads the clustered CSV in CHUNK_SIZE chunks. For each chunk,
        we encode every column's values via val_code_mapper and append
        the int32 bytes to the column's .bin file.
        We also build the B-tree month_btree by tracking contiguous month boundaries.
        We also build zone maps for 'floor_area_sqm' and 'resale_price'
        (min/max per chunk) and for 'town' (distinct codes per chunk).

        Pass 3 - Map to memmap files on disk
        Links each .bin file to a np.memmap array on disk to simulate
        disk-based storage.
        """
        logger.info("Pass 0: Sorting data by month column...")
        
        # Load and sort data to physically cluster identical months
        df = pd.read_csv(filepath)
        df['month_code'] = df['month'].apply(convert_month_str_to_code)
        df.sort_values(by='month_code', inplace=True)
        df.drop(columns=['month_code'], inplace=True)
        
        # Save clustered data to a temporary file
        sorted_filepath = os.path.join(DB_DIR, 'clustered_temp.csv')
        os.makedirs(DB_DIR, exist_ok=True)
        df.to_csv(sorted_filepath, index=False)
        filepath = sorted_filepath

        self.col_names = {col_name: i for i, col_name in enumerate(df.columns)}
        self.col_count = len(self.col_names)
        self.row_count = len(df)
        self.num_chunks = math.ceil(self.row_count / CHUNK_SIZE)

        logger.info("Pass 1: Scanning CSV to find unique values...")
        unique_vals_dict = {col: set(df[col].dropna().unique()) for col in self.col_names}
        del df # Free memory
        gc.collect()
                
        self.val_code_mapper = [None] * self.col_count
        self.code_val_mapper = [None] * self.col_count
        self.columns = [None] * self.col_count
        self.zone_maps = [[] for _ in range(self.col_count)]

        for col_name, col_idx in self.col_names.items():
            unique_list = list(unique_vals_dict[col_name])
            try:
                unique_vals = np.sort(np.array(unique_list))
            except Exception:
                unique_vals = np.array(sorted(unique_list, key=str))

            if col_name == "month":
                self.val_code_mapper[col_idx] = {val: convert_month_str_to_code(val) for val in unique_vals}
                self.code_val_mapper[col_idx] = {convert_month_str_to_code(val): val for val in unique_vals}
            elif col_name == "floor_area_sqm":
                self.val_code_mapper[col_idx] = {val: convert_floor_area_to_code(val) for val in unique_vals}
                self.code_val_mapper[col_idx] = {convert_floor_area_to_code(val): val for val in unique_vals}
            else:
                self.val_code_mapper[col_idx] = {val: idx for idx, val in enumerate(unique_vals)}
                self.code_val_mapper[col_idx] = {idx: val for idx, val in enumerate(unique_vals)}

        logger.info("Pass 2: Encoding CSV chunks, writing to disk, and building B-tree...")
        bin_filepaths = {}
        for col_name in self.col_names:
            filepath_bin = os.path.join(DB_DIR, f"{col_name}.bin")
            open(filepath_bin, 'w').close() 
            bin_filepaths[col_name] = filepath_bin

        row_offset = 0
        current_month = None
        start_row = 0

        for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE):
            for col_name, col_idx in self.col_names.items():
                encoded_chunk = chunk[col_name].map(self.val_code_mapper[col_idx]).to_numpy(dtype=np.int32)
                
                with open(bin_filepaths[col_name], 'ab') as f:
                    f.write(encoded_chunk.tobytes())
                
                if col_name == "month":
                    for i, row_month_code in enumerate(encoded_chunk):
                        global_row_idx = row_offset + i
                        if row_month_code != current_month:
                            if current_month is not None:
                                self.month_btree[int(current_month)] = (int(start_row), int(global_row_idx - 1))
                            current_month = row_month_code
                            start_row = global_row_idx
                            
                elif col_name in ["floor_area_sqm", "resale_price"]:
                    self.zone_maps[col_idx].append([int(np.min(encoded_chunk)), int(np.max(encoded_chunk))])
                elif col_name == "town":
                    self.zone_maps[col_idx].append(np.unique(encoded_chunk).tolist())           

            row_offset += len(chunk)

        if current_month is not None:
            self.month_btree[int(current_month)] = (int(start_row), int(self.row_count - 1))

        logger.info("Pass 3: Linking disk files using np.memmap...")
        for col_name, col_idx in self.col_names.items():
            self.columns[col_idx] = np.memmap(
                bin_filepaths[col_name], 
                dtype=np.int32, 
                mode='r', 
                shape=(self.row_count,)
            )

        gc.collect()
        self._log_database_state()
        logger.info(f"Loaded CSV. B-Tree index created.")