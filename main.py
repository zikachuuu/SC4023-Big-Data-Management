import csv
import numpy as np
import os
import gc
import json
import math
import logging
import time

from constants import INPUT_FILE, RESULTS_DIR, CHUNK_SIZE, TOWN_MAP_DIGIT
from columnStoreDB import ColumnStoreDB
from collections import deque
from utility import configure_logging, convert_month_year_to_code, convert_code_to_month_str, convert_floor_area_to_code


def parse_matriculation(matriculation_num):
    """
    Retrieve the target date code and target towns from the matriculation number.
    """
    # Split the matriculation number into digits
    digits = [int(d) for d in matriculation_num if d.isdigit()]
    
    # Target Year (Last Digit)
    # Rule: 5->2015 ... 9->2019, 0->2020 ... 4->2024
    last_digit = digits[-1]
    if last_digit >= 5:
        start_year = 10 + last_digit
    else:
        start_year = 20 + last_digit
        
    # Start Month (2nd Last Digit)
    # Rule: 0->Oct, 1->Jan...
    sec_last_digit = digits[-2]
    start_month = sec_last_digit if sec_last_digit != 0 else 10

    start_month_code: int = convert_month_year_to_code(start_month, start_year)
    
    # Towns (Set of all digits)
    unique_digits = set(digits)
    town_names = [TOWN_MAP_DIGIT[d] for d in unique_digits]
    
    return start_month_code, town_names


def run_queries(
        db                      : ColumnStoreDB, 
        target_start_month_code : int,
        target_town_names       : list[str], 
        matric_num              : str,
        logger                  : logging.Logger
    ):

    # ---------------------------------------------------------
    # Defining Queries
    # ---------------------------------------------------------

    # List of list to hold query in a "column store" like format. 
    #   Inner List 0: [(target start month code, target end month code) for each query]        
    #   Inner List 1: [min floor area for each query]                       
    #   Inner List 2: [(x, y) for each query] (for tracking)
    # The index of each element in the inner list corresponds to the query index
    # This allow us to scan each column in the database no more than once, and for each column, we can check the condition for all queries at once
    queries = [[], [], []]
    num_queries = 0

    for x in range(1, 9):
        for y in range(80, 151):
            num_queries += 1

            target_start_year       : int = target_start_month_code // 100  # Extract the year part from the target_start_month_code (e.g. 2001 -> 20)
            target_start_month      : int = target_start_month_code % 100   # Extract the month part from the target_start_month_code (e.g. 2001 -> 1)

            target_end_year         : int = target_start_year + (target_start_month + x - 1) // 12
            target_end_month        : int = (target_start_month + x - 1) % 12 + 1  
            target_end_month_code   : int = convert_month_year_to_code(target_end_month, target_end_year)
            target_min_floor_area   : int = y

            queries[0].append((target_start_month_code, target_end_month_code))
            queries[1].append(target_min_floor_area)
            queries[2].append((x, y))

    # List of queues to hold the valid rows for each query (one queue per query)
    # The queues are initially empty and will be populated with the row indices that satisfy the conditions of each query as we scan through the columns
    # If a row does not satify the condition of a column for a query, it will be removed from the queue for that query and will not be considered in the subsequent column scans for that query
    valid_rows: list[deque[int]] = [
        deque() for _ in range(num_queries)  
    ]


    # ------------------------------------------
    # 1. Scan month column
    # ------------------------------------------
    logger.info("Scanning month column...")
    month_col_start_time = time.time()

    month_col_idx           : int               = db.col_names["month"]             # Column index for month
    month_val_code_mapper   : dict[str, int]    = db.val_code_mapper[month_col_idx] # Original Value (e.g. "Jan-20") -> Integer Code (e.g. 2001) mapping for month column
    month_code_val_mapper   : dict[int, str]    = db.code_val_mapper[month_col_idx] # Integer Code (e.g. 2001) -> Original Value (e.g. "Jan-20") mapping for month column
    month_col               : np.ndarray        = db.columns[month_col_idx]         # Encoded month column (integer codes)
    month_zone_maps         : list[list[int]]   = db.zone_maps[month_col_idx]       # Zone maps for month column (list of [min_code, max_code] for each chunk)

    # Iterate each chunk only once
    # For each chunk, we check all queries to see which queries can be satisfied by this chunk
    for chunk_idx, (chunk_min_month, chunk_max_month) in enumerate(month_zone_maps):

        # Get the row range for this chunk
        start_row_idx           : int = chunk_idx * CHUNK_SIZE
        end_row_idx_exclusive   : int = min((chunk_idx + 1) * CHUNK_SIZE, db.row_count)

        # For each query, check if this chunk can satisfy the month condition
        for query_idx, (target_start_month, target_end_month) in enumerate(queries[0]):

            # target_start_month and target_end_month are currently encoded in the same way as that in month_val_code_mapper
            # i.e. "Jan-20" -> 2001, "Feb-20" -> 2002, ..., "Dec-20" -> 2012
            # So there is no need to use month_val_code_mapper to convert them to integer codes

            if chunk_min_month > target_end_month or chunk_max_month < target_start_month:
                # This chunk cannot satisfy the month condition for this query, skip it
                continue

            # If we reach here, this chunk may have valid rows for this query, we need to check row by row
            for row_idx in range(start_row_idx, end_row_idx_exclusive):
                row_month_code = month_col[row_idx]

                if target_start_month <= row_month_code <= target_end_month:
                    valid_rows[query_idx].append(row_idx)
    
    month_col_end_time = time.time()
    
    logger.debug("After scanning month column, valid rows for each query:")
    for query_idx in range(num_queries):
        logger.debug(f"   Query {query_idx + 1}: Start Month={queries[0][query_idx][0]}, End Month={queries[0][query_idx][1]} -> Number of Matching Rows: {len(valid_rows[query_idx])}")

    total_after_month = sum(len(rows) for rows in valid_rows)
    logger.info(f"Month scan completed in {month_col_end_time - month_col_start_time:.2f} seconds. Total candidate rows across queries: {total_after_month}")


    # -------------------------------------------
    # 2. Scan Town column
    # -------------------------------------------
    logger.info("Scanning town column...")
    town_col_start_time = time.time()

    town_col_idx        : int               = db.col_names["town"]
    town_val_code_mapper: dict[str, int]    = db.val_code_mapper[town_col_idx]
    town_code_val_mapper: dict[int, str]    = db.code_val_mapper[town_col_idx]
    town_col            : np.ndarray        = db.columns[town_col_idx]
    town_zone_maps      : list[list[int]]   = db.zone_maps[town_col_idx]

    town_codes: list[int] = [town_val_code_mapper[town] for town in target_town_names] 

    for query_idx in range(num_queries): 

        # We iterate each valid row index from the month scan for this query, 
        # check if the town of this row is in the target towns for this query, if not, we remove this row index from the queue for this query
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

    town_col_end_time = time.time()

    logger.debug("After scanning town column, valid rows for each query:")
    for query_idx in range(num_queries):
        logger.debug(f"   Query {query_idx + 1}: Towns={queries[0][query_idx][0]} -> Number of Matching Rows: {len(valid_rows[query_idx])}")

    total_after_town = sum(len(rows) for rows in valid_rows)
    logger.info(f"Town scan completed in {town_col_end_time - town_col_start_time:.2f} seconds. Total candidate rows across queries: {total_after_town}")


    # -------------------------------------------
    # 3. Scan Floor Area
    # -------------------------------------------
    logger.info("Scanning floor area column...")
    floor_area_col_start_time = time.time()

    floor_area_col_idx        : int               = db.col_names["floor_area_sqm"]
    floor_area_val_code_mapper: dict[int, int]    = db.val_code_mapper[floor_area_col_idx]
    floor_area_code_val_mapper: dict[int, int]    = db.code_val_mapper[floor_area_col_idx]
    floor_area_col            : np.ndarray        = db.columns[floor_area_col_idx]
    floor_area_zone_maps      : list[list[int]]   = db.zone_maps[floor_area_col_idx]

    for query_idx, target_min_floor_area in enumerate(queries[1]):

        # Convert the target minimum floor area to the corresponding integer code using the same encoding as that in floor_area_val_code_mapper
        # i.e. floor area 10.0 -> 100, floor area 150.0 -> 1500, etc
        min_floor_area_code = convert_floor_area_to_code(target_min_floor_area)

        # We iterate each valid row index from the town scan for this query,
        # check if the floor area of this row is greater than or equal to the target minimum floor area for this query, if not, we remove this row index from the queue for this query
        temp_queue = deque()  # Temporary queue to hold valid rows for this query after floor area scan
        while valid_rows[query_idx]:  # While there are still rows in the queue for this query

            row_idx = valid_rows[query_idx].popleft()  # Pop a row index from the queue
            row_floor_area_code = floor_area_col[row_idx]

            if row_floor_area_code >= min_floor_area_code:
                # This row is still valid for this query, keep it in temp_queue
                temp_queue.append(row_idx)
            else:
                # This row does not satisfy the floor area condition, do nothing (i.e., don't add it to temp_queue)
                pass

        # Update valid_rows[query_idx] with the rows that are still valid after floor area scan
        valid_rows[query_idx] = temp_queue
    
    floor_area_col_end_time = time.time()
    
    total_after_floor_area = sum(len(rows) for rows in valid_rows)
    logger.info(f"Floor area scan completed in {floor_area_col_end_time - floor_area_col_start_time:.2f} seconds. Total candidate rows across queries: {total_after_floor_area}") 
                
    logger.info(f"Writing detailed query results for matriculation number {matric_num} to log file.")
    logger.debug(f"Query Results for Matriculation Number {matric_num}:")

    for query_idx in range(len(queries[0])):  # Assuming all inner lists in queries have the same length
        x               : int = queries[2][query_idx][0]
        y               : int = queries[2][query_idx][1]
        start_month_str : str = convert_code_to_month_str(queries[0][query_idx][0])
        end_month_str   : str = convert_code_to_month_str(queries[0][query_idx][1])
        min_floor_area  : int = queries[1][query_idx]

        logger.debug(f"Query {query_idx + 1} (x={x}, y={y}): Start Month={start_month_str}, End Month={end_month_str}, Min Floor Area={min_floor_area} -> Number of Matching Rows: {len(valid_rows[query_idx])}")

    logger.info("All query scans completed.")
    logger.info("Selecting row with least psm among the valid rows for each query and writing results to log file...")


    # -------------------------------------------
    # 4. Select row with least psm
    # -------------------------------------------

    # For each query, we select the row with the least psm among the valid rows for this query
    # Where psm = resale_price / floor_area

    # List to hold (row_idx, psm) for the row with least psm for each query
    # If there is no valid row for a query, we will store (None, None) for that query
    min_psm_results: list[tuple[int | None, float | None]] = []  

    resale_price_col_idx        : int               = db.col_names["resale_price"]
    resale_price_val_code_mapper: dict[int, int]    = db.val_code_mapper[resale_price_col_idx]
    resale_price_code_val_mapper: dict[int, int]    = db.code_val_mapper[resale_price_col_idx]
    resale_price_col            : np.ndarray        = db.columns[resale_price_col_idx]
    resale_price_zone_maps      : list[list[int]]   = db.zone_maps[resale_price_col_idx]

    for query_idx in range(num_queries):
        
        min_psm = math.inf
        min_psm_row_idx = None

        for row_idx in valid_rows[query_idx]:

            resale_price    : int   = resale_price_code_val_mapper[resale_price_col[row_idx]]  
            floor_area      : float = floor_area_code_val_mapper[floor_area_col[row_idx]] 
            psm = resale_price / floor_area

            if psm < min_psm:
                min_psm = psm
                min_psm_row_idx = row_idx

        if min_psm_row_idx is not None:
            min_psm_results.append((min_psm_row_idx, min_psm))
        else:
            min_psm_results.append((None, None))

    logger.info("Selection of rows with least psm completed. Writing final results to CSV file...")

    # Save to ScanResult_<matriculation_number>.csv in Results folder
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file_path = os.path.join(RESULTS_DIR, f"ScanResult_{matric_num}.csv")

    with open(results_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["(x,y)", "Year", "Month", "Town", "Block", "Floor_Area", "Flat_Model", 'Lease_Commence_Date', "Price_Per_Square_Meter"])

        for query_idx, (min_psm_row_idx, min_psm) in enumerate(min_psm_results):
            x: int = queries[2][query_idx][0]
            y: int = queries[2][query_idx][1]

            
            if min_psm_row_idx is not None:

                block_code_val_mapper: dict[int, str] = db.code_val_mapper[db.col_names["block"]]
                block_col: np.ndarray = db.columns[db.col_names["block"]]

                flat_model_code_val_mapper: dict[int, str] = db.code_val_mapper[db.col_names["flat_model"]]
                flat_model_col: np.ndarray = db.columns[db.col_names["flat_model"]]

                lease_commence_date_code_val_mapper: dict[int, int] = db.code_val_mapper[db.col_names["lease_commence_date"]]
                lease_commence_date_col: np.ndarray = db.columns[db.col_names["lease_commence_date"]]

                year                : str   = "20" + str(month_col[min_psm_row_idx])[:2]   # Extract the year part from the month code (e.g. 2001 -> "20") and add "20" in front to get the full year string (e.g. "20" + "20" -> "2020")
                month               : str   = str (month_col[min_psm_row_idx])[2:]   # Extract the month part from the month code (e.g. 2001 -> "01")
                town                : str   = town_code_val_mapper[town_col[min_psm_row_idx]]
                block               : str   = block_code_val_mapper[block_col[min_psm_row_idx]]
                floor_area          : float = floor_area_code_val_mapper[floor_area_col[min_psm_row_idx]]
                flat_model          : str   = flat_model_code_val_mapper[flat_model_col[min_psm_row_idx]]
                lease_commence_date : int   = lease_commence_date_code_val_mapper[lease_commence_date_col[min_psm_row_idx]]

                csv_writer.writerow([f"({x},{y})", year, month, town, block, floor_area, flat_model, lease_commence_date, int(min_psm)])
            else:
                csv_writer.writerow([f"({x},{y})", "No Result", "No Result", "No Result", "No Result", "No Result", "No Result", "No Result", "No Result"])

    logger.info(f"Results written to {results_file_path}")


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":

    matriculation_number = input("Enter your matriculation number (e.g. A0123456B): ")

    logger  : logging.Logger    = configure_logging(matriculation_number)
    db      : ColumnStoreDB     = ColumnStoreDB()

    logger.info("Database initialized.")
    logger.info("Loading CSV into Column Store Database...")

    if os.path.exists(INPUT_FILE):
        db.load_csv(INPUT_FILE)

        logger.info("Database loaded.")

        (start_month_code, town_names) = parse_matriculation(matriculation_number)

        start_month_str: str = convert_code_to_month_str(start_month_code)

        logger.info(f"Parsed Matriculation: Start Month={start_month_str}, Towns={town_names}")
        logger.info("Running queries for x in [1, 8], y in [80, 150]...")

        run_queries(db, start_month_code, town_names, matriculation_number, logger)
        
        del db
        gc.collect()
        logger.info("Run completed and memory cleaned up.")
        
    else:
        logger.error(f"{INPUT_FILE} not found.")