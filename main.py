import csv
import numpy as np
import os
import gc
import math
import logging
import time
from collections import deque

from constants import INPUT_FILE, RESULTS_DIR, CHUNK_SIZE, TOWN_MAP_DIGIT, MAX_PSM
from columnStoreDB import ColumnStoreDB
from utility import configure_logging, convert_month_year_to_code, convert_code_to_month_str, convert_floor_area_to_code

def parse_matriculation(matriculation_num):
    digits = [int(d) for d in matriculation_num if d.isdigit()]
    
    last_digit = digits[-1]
    if last_digit >= 5: start_year = 10 + last_digit
    else: start_year = 20 + last_digit
        
    sec_last_digit = digits[-2]
    start_month = sec_last_digit if sec_last_digit != 0 else 10

    start_month_code: int = convert_month_year_to_code(start_month, start_year)
    
    unique_digits = set(digits)
    town_names = [TOWN_MAP_DIGIT[d] for d in unique_digits]
    
    return start_month_code, town_names

def run_queries(db: ColumnStoreDB, target_start_month_code: int, target_town_names: list[str], matric_num: str, logger: logging.Logger):
    
    town_col_idx = db.col_names["town"]
    town_col = db.columns[town_col_idx]
    town_codes = [db.val_code_mapper[town_col_idx][t] for t in target_town_names]
    town_zone_maps = db.zone_maps[town_col_idx]

    floor_area_col_idx = db.col_names["floor_area_sqm"]
    floor_area_col = db.columns[floor_area_col_idx]
    floor_area_zone_maps = db.zone_maps[floor_area_col_idx]

    queries = [[], [], []]
    valid_rows = []

    for x in range(1, 9):
        for y in range(80, 151):
            target_start_year = target_start_month_code // 100 
            target_start_month = target_start_month_code % 100  
            target_end_year = target_start_year
            target_end_month = target_start_month + x - 1
            if target_end_month > 12:
                target_end_month -= 12
                target_end_year += 1

            queries[0].append((target_start_month_code, convert_month_year_to_code(target_end_month, target_end_year)))
            queries[1].append(y)
            queries[2].append((x, y))
            valid_rows.append(deque())

    logger.info("Executing Queries via Clustered IOBTree -> Zone Maps -> NumPy Vectorization...")
    query_start_time = time.time()

    # Query Start
    for query_idx in range(len(queries[0])):
        target_start_month, target_end_month = queries[0][query_idx]
        target_min_floor_area = convert_floor_area_to_code(queries[1][query_idx])

        # 1. Check Clustered B-Tree Index on Month
        month_ranges = list(db.month_btree.values(min=target_start_month, max=target_end_month))
        if not month_ranges:
            continue # No matching months found at all

        min_row_idx = month_ranges[0][0]
        max_row_idx = month_ranges[-1][1]

        start_chunk = min_row_idx // CHUNK_SIZE
        end_chunk = max_row_idx // CHUNK_SIZE

        # 2 & 3. Check Zone Maps over the affected chunks
        for chunk_idx in range(start_chunk, end_chunk + 1):
            
            # Check Town Zone Map
            chunk_towns = town_zone_maps[chunk_idx]
            if not any(t in chunk_towns for t in town_codes):
                continue

            # Check Floor Area Zone Map
            chunk_min_area, chunk_max_area = floor_area_zone_maps[chunk_idx]
            if chunk_max_area < target_min_floor_area:
                continue

            # 4. If Zone Maps pass, perform Vectorized Column-Oriented Processing
            chunk_start_row = max(chunk_idx * CHUNK_SIZE, min_row_idx)
            chunk_end_row = min((chunk_idx + 1) * CHUNK_SIZE, db.row_count, max_row_idx + 1)

            # Extract the column slices for this specific chunk
            town_slice = town_col[chunk_start_row:chunk_end_row]
            floor_area_slice = floor_area_col[chunk_start_row:chunk_end_row]

            # Vectorized evaluation on the entire town column slice
            valid_town_mask = np.isin(town_slice, town_codes)

            # Vectorized evaluation on the entire floor area column slice
            valid_area_mask = floor_area_slice >= target_min_floor_area

            # Combine masks using bitwise AND
            combined_mask = valid_town_mask & valid_area_mask

            # Extract the local row indices where the combined mask is True
            valid_local_indices = np.where(combined_mask)[0]

            # Translate local chunk indices back to global row indices
            valid_global_indices = valid_local_indices + chunk_start_row

            # Extend the deque with the valid global row indices
            valid_rows[query_idx].extend(valid_global_indices)

    logger.info(f"Query Execution completed in {time.time() - query_start_time:.2f} seconds.")
    logger.info("Selecting row with least psm among the valid rows for each query and writing results to log file...")

    # -------------------------------------------
    # Select row with least psm
    # -------------------------------------------
    min_psm_results: list[tuple[int | None, float | None]] = []  

    resale_price_col_idx = db.col_names["resale_price"]
    resale_price_code_val_mapper = db.code_val_mapper[resale_price_col_idx]
    resale_price_col = db.columns[resale_price_col_idx]
    
    floor_area_code_val_mapper = db.code_val_mapper[floor_area_col_idx]

    month_col_idx = db.col_names["month"]
    month_col = db.columns[month_col_idx]
    
    town_code_val_mapper = db.code_val_mapper[town_col_idx]
    
    block_col_idx = db.col_names["block"]
    block_code_val_mapper = db.code_val_mapper[block_col_idx]
    block_col = db.columns[block_col_idx]

    for query_idx in range(len(queries[0])):
        min_psm = math.inf
        min_psm_row_idx = None

        for row_idx in valid_rows[query_idx]:
            resale_price = resale_price_code_val_mapper[resale_price_col[row_idx]]  
            floor_area = floor_area_code_val_mapper[floor_area_col[row_idx]] 
            psm = resale_price / floor_area

            if psm <= MAX_PSM:
                if psm < min_psm:
                    min_psm = psm
                    min_psm_row_idx = row_idx
                # We need to apply tiebreakers if we have multiple rows with the same minimum psm.
                elif psm == min_psm:
                    curr_month_code = month_col[row_idx]
                    min_month_code = month_col[min_psm_row_idx]
                    
                    if curr_month_code < min_month_code:
                        min_psm_row_idx = row_idx
                    elif curr_month_code == min_month_code:
                        curr_town = town_code_val_mapper[town_col[row_idx]]
                        min_town = town_code_val_mapper[town_col[min_psm_row_idx]]
                        
                        if curr_town < min_town:
                            min_psm_row_idx = row_idx
                        elif curr_town == min_town:
                            curr_block = block_code_val_mapper[block_col[row_idx]]
                            min_block = block_code_val_mapper[block_col[min_psm_row_idx]]
                            
                            if curr_block < min_block:
                                min_psm_row_idx = row_idx

        if min_psm_row_idx is not None and min_psm <= MAX_PSM:
            min_psm_results.append((min_psm_row_idx, min_psm))
        else:
            min_psm_results.append((None, None))

    logger.info("Selection of rows with least psm completed. Writing final results to CSV file...")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file_path = os.path.join(RESULTS_DIR, f"ScanResult_{matric_num}.csv")

    with open(results_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["(x,y)", "Year", "Month", "Town", "Block", "Floor_Area", "Flat_Model", 'Lease_Commence_Date', "Price_Per_Square_Meter"])

        for query_idx, (min_psm_row_idx, min_psm) in enumerate(min_psm_results):
            x = queries[2][query_idx][0]
            y = queries[2][query_idx][1]
            
            if min_psm_row_idx is not None:
                flat_model_code_val_mapper = db.code_val_mapper[db.col_names["flat_model"]]
                flat_model_col = db.columns[db.col_names["flat_model"]]

                lease_commence_date_code_val_mapper = db.code_val_mapper[db.col_names["lease_commence_date"]]
                lease_commence_date_col = db.columns[db.col_names["lease_commence_date"]]

                year = "20" + str(month_col[min_psm_row_idx])[:2]
                month = str(month_col[min_psm_row_idx])[2:]
                town = town_code_val_mapper[town_col[min_psm_row_idx]]
                block = block_code_val_mapper[block_col[min_psm_row_idx]]
                floor_area = floor_area_code_val_mapper[floor_area_col[min_psm_row_idx]]
                flat_model = flat_model_code_val_mapper[flat_model_col[min_psm_row_idx]]
                lease_commence_date = lease_commence_date_code_val_mapper[lease_commence_date_col[min_psm_row_idx]]

                csv_writer.writerow([f"({x},{y})", year, month, town, block, floor_area, flat_model, lease_commence_date, int(min_psm)])
            else:
                csv_writer.writerow([f"({x},{y})", "No Result", "No Result", "No Result", "No Result", "No Result", "No Result", "No Result", "No Result"])

    logger.info(f"Results written to {results_file_path}")

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    matriculation_number = input("Enter your matriculation number (e.g. A0123456B): ")

    logger = configure_logging(matriculation_number)
    db = ColumnStoreDB()

    logger.info("Database initialized.")
    logger.info("Loading CSV into Column Store Database...")

    if os.path.exists(INPUT_FILE):
        db.load_csv(INPUT_FILE)
        logger.info("Database loaded.")

        (start_month_code, town_names) = parse_matriculation(matriculation_number)
        start_month_str = convert_code_to_month_str(start_month_code)

        logger.info(f"Parsed Matriculation: Start Month={start_month_str}, Towns={town_names}")
        logger.info("Running queries for x in [1, 8], y in [80, 150]...")

        run_queries(db, start_month_code, town_names, matriculation_number, logger)
        
        del db
        gc.collect()
        logger.info("Run completed and memory cleaned up.")
        
    else:
        logger.error(f"{INPUT_FILE} not found.")