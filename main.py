import csv
import numpy as np
import os
import gc
import json
import math

from constants import INPUT_FILE, CHUNK_SIZE, TOWN_MAP_DIGIT
from columnStoreDB import ColumnStoreDB
from collections import deque


def parse_matriculation(matriculation_num):
    """
    Retrieve the target start year, start month, and target towns from the matriculation number.
    """

    # Split the matriculation number into digits
    digits = [int(d) for d in matriculation_num if d.isdigit()]
    
    # 1. Target Year (Last Digit)
    # Rule: 5->2015 ... 9->2019, 0->2020 ... 4->2024
    last_digit = digits[-1]
    if last_digit >= 5:
        target_year = 10 + last_digit
    else:
        target_year = 20 + last_digit
        
    # 2. Start Month (2nd Last Digit)
    # Rule: 0->Oct, 1->Jan...
    sec_last_digit = digits[-2]
    start_month = sec_last_digit if sec_last_digit != 0 else 10
    
    # 3. Towns (Set of all digits)
    unique_digits = set(digits)
    town_names = [TOWN_MAP_DIGIT[d] for d in unique_digits]
    
    return target_year, start_month, town_names



def run_queries(db, target_start_year, target_start_month, target_town_names, matric_num):

    # List of list to hold query in a "column store" like format. 
    #   Inner List 0: [(start month, end month) for each query]
    #   Inner List 1: [(start year, end year) for each query]
    #   Inner List 2: [min floor area for each query] 
    # The index of the inner list corresponds to the query index
    # This allow us to do shared scan (process all queries in one pass)
    queries = [[], [], []]
    num_queries = 0

    for x in range(1, 9):
        for y in range(80, 151):
            num_queries += 1

            target_end_year = target_start_year + (target_start_month + x - 1) // 12
            target_end_month = (target_start_month + x - 1) % 12 + 1  
            target_min_floor_area = y

            queries[0].append((target_start_month, target_end_month))
            queries[1].append((target_start_year, target_end_year))
            queries[2].append(target_min_floor_area)

    
    # ---------------------------------------------------------
    # Scanning: We scan each column once only
    # ---------------------------------------------------------

    # List of queues to hold the valid rows for each query (empty at the start)
    valid_rows: list[deque[int]] = [
        deque() for _ in range(num_queries)  # Assuming all inner lists in queries have the same length
    ]

    # ------------------------------------------
    # 1. Scan month column
    # ------------------------------------------
    month_col_idx           : int               = db.col_names["month"]             # Column index for month
    month_val_code_mapper   : dict[str, int]    = db.val_code_mapper[month_col_idx] # Original Value (e.g. "Jan") -> Integer Code mapping for month column
    month_col               : np.ndarray        = db.columns[month_col_idx]         # Encoded month column (integer codes)
    month_zone_maps         : list[list[int]]   = db.zone_maps[month_col_idx]       # Zone maps for month column (list of [min_code, max_code] for each chunk)

    for query_idx, (target_start_month, target_end_month) in enumerate(queries[0]):

        # month and month code are actually the same (1 to 12)]
        # so val_code_mapper is not used

        # Check against zone maps to quickly eliminate chunks
        for chunk_idx, (chunk_min_month, chunk_max_month) in enumerate(month_zone_maps):

            if chunk_max_month < target_start_month or chunk_min_month > target_end_month:
                # This chunk cannot satisfy the month condition, skip it
                continue

            # If we reach here, this chunk may have valid rows for this query, we need to check row by row
            start_row_idx           : int = chunk_idx * CHUNK_SIZE
            end_row_idx_exclusive   : int = min((chunk_idx + 1) * CHUNK_SIZE, db.row_count)
    
            for row_idx in range(start_row_idx, end_row_idx_exclusive):
                month_code = month_col[row_idx]
                if target_start_month <= month_code <= target_end_month:
                    valid_rows[query_idx].append(row_idx)
    
    print (f"After scanning month column, valid rows for each query: ")
    for query_idx in range(num_queries):
        print (f"   Query {query_idx + 1}: Start Month={queries[0][query_idx][0]}, End Month={queries[0][query_idx][1]} -> Number of Matching Rows: {len(valid_rows[query_idx])}")

    # -------------------------------------------
    # 2. Scan year column
    # -------------------------------------------
    year_col_idx        : int               = db.col_names["year"]
    year_val_code_mapper: dict[int, int]    = db.val_code_mapper[year_col_idx]
    year_col            : np.ndarray        = db.columns[year_col_idx]
    year_zone_maps      : list[list[int]]   = db.zone_maps[year_col_idx]

    for query_idx, (target_start_year, target_end_year) in enumerate(queries[1]):

        start_year_code = year_val_code_mapper[target_start_year]
        end_year_code   = year_val_code_mapper[target_end_year]

        # Check against zone maps to quickly eliminate chunks
        # We skip this chunk if 
        #   1) it does not contain any valid rows for this query (from month scan), or 
        #   2) it cannot satisfy the year condition based on the zone map
        for chunk_idx, (chunk_min_month, chunk_max_month) in enumerate(year_zone_maps):

            # Skip this chunk if it does not contain the valid rows (from month scan) for this query
            start_row_idx           : int = chunk_idx * CHUNK_SIZE
            end_row_idx_exclusive   : int = min((chunk_idx + 1) * CHUNK_SIZE, db.row_count)

            if not any (start_row_idx <= row_idx < end_row_idx_exclusive for row_idx in valid_rows[query_idx]):
                # No valid rows for this query in this chunk, skip it
                continue

            if chunk_max_month < start_year_code or chunk_min_month > end_year_code:
                # This chunk cannot satisfy the year condition, skip it
                continue
        
            # If we reach here, this chunk may have valid rows for this query, we need to check row by row
            temp_queue = deque()  # Temporary queue to hold valid rows for this query after year scan

            while valid_rows[query_idx]:  # While there are still rows in the queue for this query
                row_idx = valid_rows[query_idx].popleft()  # Pop a row index from the queue
                year_code = year_col[row_idx]
                if start_year_code <= year_code <= end_year_code:
                    # This row is still valid for this query, keep it in temp_queue
                    temp_queue.append(row_idx)
                    continue

                else:
                    # This row does not satisfy the year condition, remove it from valid rows for this query
                    pass  # Do nothing, since we're using temp_queue to hold valid rows

            # Update valid_rows[query_idx] with the rows that are still valid after year scan
            valid_rows[query_idx] = temp_queue

    print (f"After scanning year column, valid rows for each query: {[len(rows) for rows in valid_rows]}")

    # -------------------------------------------
    # 3. Scan Town column
    # -------------------------------------------
    town_col_idx        : int               = db.col_names["town"]
    town_val_code_mapper: dict[str, int]    = db.val_code_mapper[town_col_idx]
    town_col            : np.ndarray        = db.columns[town_col_idx]
    town_zone_maps      : list[list[int]]   = db.zone_maps[town_col_idx]

    town_codes: list[int] = [town_val_code_mapper[town] for town in target_town_names] 

    for query_idx in range(len(queries[0])):  # Assuming all inner lists in queries have the same length

        # Check against zone maps to quickly eliminate chunks
        # We skip this chunk if 
        #   1) it does not contain any valid rows for this query (from previous scans), or 
        #   2) it cannot satisfy the town condition based on the zone map
        for chunk_idx, chunk_towns in enumerate(town_zone_maps):

            # Skip this chunk if it does not contain the valid rows (from previous scans) for this query
            start_row_idx           : int = chunk_idx * CHUNK_SIZE
            end_row_idx_exclusive   : int = min((chunk_idx + 1) * CHUNK_SIZE, db.row_count)

            if not any (start_row_idx <= row_idx < end_row_idx_exclusive for row_idx in valid_rows[query_idx]):
                # No valid rows for this query in this chunk, skip it
                continue

            if not any (town_code in town_codes for town_code in chunk_towns):
                # This chunk does not contain any of the target towns, skip it
                continue
        
            # If we reach here, this chunk may have valid rows for this query, we need to check row by row
            temp_queue = deque()  # Temporary queue to hold valid rows for this query after town scan

            while valid_rows[query_idx]:  # While there are still rows in the queue for this query
                row_idx = valid_rows[query_idx].popleft()  # Pop a row index from the queue
                town_code = town_col[row_idx]
                if town_code in town_codes:
                    # This row is still valid for this query, keep it in temp_queue
                    temp_queue.append(row_idx)
                else:
                    # This row does not satisfy the town condition, do nothing (i.e., don't add it to temp_queue)
                    pass

            # Update valid_rows[query_idx] with the rows that are still valid after town scan
            valid_rows[query_idx] = temp_queue

    print (f"After scanning town column, valid rows for each query: {[len(rows) for rows in valid_rows]}")

    # -------------------------------------------
    # 4. Scan Floor Area
    # -------------------------------------------
    floor_area_col_idx        : int               = db.col_names["floor_area_sqm"]
    floor_area_val_code_mapper: dict[int, int]    = db.val_code_mapper[floor_area_col_idx]
    floor_area_col            : np.ndarray        = db.columns[floor_area_col_idx]
    floor_area_zone_maps      : list[list[int]]   = db.zone_maps[floor_area_col_idx]

    for query_idx, target_min_floor_area in enumerate(queries[2]):
        
        # Abit tricky here because the given floor area may not exist in the dataset
        # So we need to find its next larger and next smaller floor area in the dataset
        # and give this floor area a mean code between the codes of these two floor areas
        # ie if next smaller is code 10 and next larger is code 11, then we give this floor area code 10.5
        # This way, we can still use the zone map to do quick elimination based on this

        if target_min_floor_area in floor_area_val_code_mapper:
            min_floor_area_code = floor_area_val_code_mapper[target_min_floor_area]
        else:
            # Find next smaller and next larger floor area in the dataset
            smaller_areas = [area for area in floor_area_val_code_mapper if area < target_min_floor_area]
            larger_areas  = [area for area in floor_area_val_code_mapper if area > target_min_floor_area]

            if not smaller_areas:
                next_smaller_code = -math.inf  # If no smaller area, set code to negative infinity
            else:
                next_smaller_area = max(smaller_areas)
                next_smaller_code = floor_area_val_code_mapper[next_smaller_area]

            if not larger_areas:
                next_larger_code = math.inf  # If no larger area, set code to infinity
            else:
                next_larger_area = min(larger_areas)
                next_larger_code = floor_area_val_code_mapper[next_larger_area]

            min_floor_area_code = (next_smaller_code + next_larger_code) / 2

        # Check against zone maps to quickly eliminate chunks
        # We skip this chunk if 
        #   1) it does not contain any valid rows for this query (from previous scans), or 
        #   2) it cannot satisfy the floor area condition based on the zone map
        for chunk_idx, (chunk_min_month, chunk_max_month) in enumerate(floor_area_zone_maps):

            # Skip this chunk if it does not contain the valid rows (from previous scans) for this query
            start_row_idx           : int = chunk_idx * CHUNK_SIZE
            end_row_idx_exclusive   : int = min((chunk_idx + 1) * CHUNK_SIZE, db.row_count)

            if not any (start_row_idx <= row_idx < end_row_idx_exclusive for row_idx in valid_rows[query_idx]):
                # No valid rows for this query in this chunk, skip it
                continue

            if chunk_max_month < min_floor_area_code:
                # This chunk cannot satisfy the floor area condition, skip it
                continue
        
            # If we reach here, this chunk may have valid rows for this query, we need to check row by row
            temp_queue = deque()  # Temporary queue to hold valid rows for this query after floor area scan

            while valid_rows[query_idx]:  # While there are still rows in the queue for this query
                row_idx = valid_rows[query_idx].popleft()  # Pop a row index from the queue
                floor_area_code = floor_area_col[row_idx]
                if floor_area_code >= min_floor_area_code:
                    # This row is still valid for this query, keep it in temp_queue
                    temp_queue.append(row_idx)
                else:
                    # This row does not satisfy the floor area condition, remove it from valid rows for this query
                    pass  # Do nothing, since we're using temp_queue to hold valid rows

            # Update valid_rows[query_idx] with the rows that are still valid after floor area scan
            valid_rows[query_idx] = temp_queue

    
    print (f"Query Results for Matriculation Number {matric_num}:")
    for query_idx in range(len(queries[0])):  # Assuming all inner lists in queries have the same length
        print (f"Query {query_idx + 1}: Start Month={queries[0][query_idx][0]}, End Month={queries[0][query_idx][1]}, Start Year={queries[1][query_idx][0]}, End Year={queries[1][query_idx][1]}, Min Floor Area={queries[2][query_idx]} -> Number of Matching Rows: {len(valid_rows[query_idx])}")



# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    db = ColumnStoreDB()
    print ("Database Initialized.")
    print ("Loading CSV into Column Store Database...")

    if os.path.exists(INPUT_FILE):
        db.load_csv(INPUT_FILE)

        print ("Database loaded.")
        matriculation_number = input("Enter your matriculation number (e.g. A0123456B): ")
        (start_year, start_month, town_names) = parse_matriculation(matriculation_number)

        print (f"Parsed Matriculation: Start Year={start_year}, Start Month={start_month}, Towns={town_names}")
        print ("Running Queries for x in [1, 8], y in [80, 150]...")

        run_queries(db, start_year, start_month, town_names, matriculation_number)
        
        del db
        gc.collect()
        
    else:
        print(f"Error: {INPUT_FILE} not found.")