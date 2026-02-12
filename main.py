import csv
import numpy as np
import os
import gc

# ---------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------

INPUT_FILE = os.path.join(os.path.dirname(__file__), 'ResalePricesSingapore.csv')
CHUNK_SIZE = 1000  # Size for Zone Map chunks

# Mapping from Matric Digit to Town (Source: Table 1)
TOWN_MAP_DIGIT = {
    0: "BEDOK", 1: "BUKIT PANJANG", 2: "CLEMENTI", 3: "CHOA CHU KANG", 4: "HOUGANG",
    5: "JURONG WEST", 6: "PASIR RIS", 7: "TAMPINES", 8: "WOODLANDS", 9: "YISHUN"
}

# Mapping for Matric Month (Source: Description)
# 1->Jan, 2->Feb... 9->Sep, 0->Oct
MONTH_MAP_DIGIT = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 0: 10
}

# ---------------------------------------------------------
# COLUMN STORE CLASS
# ---------------------------------------------------------

class ColumnStoreDB:
    def __init__(self):
        self.columns = {} # Stores numpy arrays: name -> array
        self.dictionaries = {} # Stores lookup tables for encoding: name -> list
        self.row_count = 0
        self.headers = []
        self.zone_maps = {} # name -> [{'min': val, 'max': val}, ...]

    def load_csv(self, filepath):
        print(f"Loading {filepath}...")
        data_buffer = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            self.headers = next(reader)
            
            # Initialize lists for each column
            for h in self.headers:
                data_buffer[h] = []
            
            # Read rows
            for row in reader:
                for i, val in enumerate(row):
                    data_buffer[self.headers[i]].append(val)
                    
        self.row_count = len(data_buffer[self.headers[0]])
        print(f"Loaded {self.row_count} rows.")

        # Convert to initial numpy arrays (as strings/objects first)
        for h in self.headers:
            self.columns[h] = np.array(data_buffer[h])
        
        # Clean up buffer
        del data_buffer
        gc.collect()

    def optimize(self):
        print("Starting optimizations...")
        
        # --- 1. Dictionary Encoding (Compression) ---
        # We need to encode strings to integers. 
        # Crucial: Dictionaries must be sorted to allow range queries if needed later.
        
        # Columns to encode (Strings)
        enc_cols = ['month', 'town', 'flat_type', 'block', 'street_name', 'flat_model', 'storey_range']
        
        for col in enc_cols:
            unique_vals = np.unique(self.columns[col])
            # np.unique returns sorted unique elements
            self.dictionaries[col] = unique_vals
            
            # Map values to integers
            # np.searchsorted finds indices where elements should be inserted to maintain order -> efficient mapping
            self.columns[col] = np.searchsorted(unique_vals, self.columns[col]).astype(np.int32)
            
        # Convert numeric columns explicitly
        self.columns['floor_area'] = self.columns['floor_area'].astype(np.float32)
        self.columns['resale_price'] = self.columns['resale_price'].astype(np.float32)
        # Lease_Commence_Date is purely int in the file usually
        self.columns['lease_commence_date'] = self.columns['lease_commence_date'].astype(np.int32)

        # --- 2. Sorting ---
        # Sort order: Month -> Town -> Floor Area -> Resale Price
        print("Sorting data...")
        
        # lexsort sorts by keys in reverse order (last key is primary)
        sort_indices = np.lexsort((
            self.columns['resale_price'],
            self.columns['floor_area'],
            self.columns['town'],
            self.columns['month']
        ))
        
        # Reorder ALL columns based on sort_indices
        for col in self.headers:
            self.columns[col] = self.columns[col][sort_indices]
            
        # --- 3. Zone Maps ---
        # Create min/max metadata for chunks
        print("Building Zone Maps...")
        num_chunks = (self.row_count + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        # We only strictly need zone maps for filter columns: month, town, floor_area
        zone_cols = ['month', 'town', 'floor_area']
        
        for col in zone_cols:
            self.zone_maps[col] = []
            arr = self.columns[col]
            for i in range(num_chunks):
                start = i * CHUNK_SIZE
                end = min((i + 1) * CHUNK_SIZE, self.row_count)
                chunk = arr[start:end]
                self.zone_maps[col].append({
                    'min': np.min(chunk),
                    'max': np.max(chunk)
                })

    def decode_value(self, col_name, encoded_val):
        """Helper to get original string back from int code"""
        if col_name in self.dictionaries:
            return self.dictionaries[col_name][encoded_val]
        return encoded_val

    def get_month_int(self, year_str, month_str):
        """Converts 'YYYY', 'MM' inputs into the encoded integer for the 'month' column"""
        s = f"{year_str}-{str(month_str).zfill(2)}"
        # Find index in dictionary
        idx = np.searchsorted(self.dictionaries['month'], s)
        if idx < len(self.dictionaries['month']) and self.dictionaries['month'][idx] == s:
            return idx
        return -1 # Not found

# ---------------------------------------------------------
# QUERY LOGIC
# ---------------------------------------------------------

def parse_matric(matric):
    """
    Parses matric number based on rules.
    Format: A<7 digits>B
    """
    digits = [int(c) for c in matric if c.isdigit()]
    if len(digits) != 7:
        raise ValueError("Invalid Matriculation Number: Must have 7 digits.")
    
    # 1. Target Year (Last Digit)
    # Rule: 5->2015 ... 9->2019, 0->2020 ... 4->2024
    last_digit = digits[-1]
    if last_digit >= 5:
        target_year = 2010 + last_digit
    else:
        target_year = 2020 + last_digit
        
    # 2. Start Month (2nd Last Digit)
    # Rule: 0->Oct, 1->Jan...
    sec_last_digit = digits[-2]
    start_month = MONTH_MAP_DIGIT.get(sec_last_digit, 1) # default to Jan if issue
    
    # 3. Towns (Set of all digits)
    target_towns_idx = []
    unique_digits = set(digits)
    
    # We need to map these to the DB's internal encoding for Towns
    # First get string names
    town_names = [TOWN_MAP_DIGIT[d] for d in unique_digits if d in TOWN_MAP_DIGIT]
    
    return target_year, start_month, town_names

def run_queries(db, matric_num):
    print(f"\nProcessing Matric: {matric_num}")
    
    try:
        start_year, start_month, town_names = parse_matric(matric_num)
    except Exception as e:
        print(f"Error parsing matric: {e}")
        return

    print(f"Target Start: {start_year}-{start_month:02d}")
    print(f"Target Towns: {town_names}")

    # Convert Town Names to DB Encoded Integers
    town_encoded_ids = []
    for t_name in town_names:
        # Find index in the Town dictionary
        # Dictionary is sorted, can use searchsorted or manual check
        idx = np.searchsorted(db.dictionaries['town'], t_name)
        if idx < len(db.dictionaries['town']) and db.dictionaries['town'][idx] == t_name:
            town_encoded_ids.append(idx)
    town_encoded_ids.sort() # Keep sorted for efficiency

    results = []
    log_entries = []

    # Loop x (Duration in months) and y (Min Floor Area)
    for x in range(1, 9):
        for y in range(80, 151):
            
            # --- CALCULATE DATE RANGE ---
            # End date calculation logic
            # Current date is year, month. Add x-1 months.
            total_months = start_month + (x - 1)
            end_y = start_year + (total_months - 1) // 12
            end_m = (total_months - 1) % 12 + 1
            
            # Convert start/end to encoded integers
            # We need to find the range of encoded values that correspond to this date range
            # Format in DB is YYYY-MM
            s_str = f"{start_year}-{start_month:02d}"
            e_str = f"{end_y}-{end_m:02d}"
            
            # Find integer range in the dictionary
            # Note: Because 'Month' dictionary is sorted strings "2015-01", "2015-02"... 
            # we can just find the start string index and end string index.
            
            start_code = np.searchsorted(db.dictionaries['month'], s_str)
            end_code = np.searchsorted(db.dictionaries['month'], e_str, side='right') - 1
            
            # Handle case where generated date exceeds dataset (e.g. 2026)
            if start_code >= len(db.dictionaries['month']):
                continue # Range is completely outside

            # --- SCANNING (The Filter Pipeline) ---
            
            # 1. Identify Candidate Chunks (Zone Map Optimization)
            valid_chunk_indices = []
            num_chunks = len(db.zone_maps['month'])
            
            for i in range(num_chunks):
                # Check Month overlap
                m_min = db.zone_maps['month'][i]['min']
                m_max = db.zone_maps['month'][i]['max']
                if not (m_max < start_code or m_min > end_code):
                    # Check Area overlap (Optimization: early prune if max area < y)
                    a_max = db.zone_maps['floor_area'][i]['max']
                    if a_max >= y:
                        valid_chunk_indices.append(i)
            
            matched_indices = []
            
            # 2. Scan Columns in Valid Chunks
            for chunk_idx in valid_chunk_indices:
                idx_start = chunk_idx * CHUNK_SIZE
                idx_end = min((chunk_idx + 1) * CHUNK_SIZE, db.row_count)
                
                # Retrieve slice for this chunk
                chunk_months = db.columns['month'][idx_start:idx_end]
                
                # A. Filter Month
                # Get local indices where month is in range
                # Using simple numpy boolean masking
                mask = (chunk_months >= start_code) & (chunk_months <= end_code)
                
                if not np.any(mask):
                    continue
                
                # Get global indices of survivors
                survivors = np.where(mask)[0] + idx_start
                
                # B. Filter Town (on survivors)
                # Retrieve Town values for survivors
                current_towns = db.columns['town'][survivors]
                # Check if town is in our target list
                # np.isin is efficient for this
                town_mask = np.isin(current_towns, town_encoded_ids)
                
                if not np.any(town_mask):
                    continue
                    
                survivors = survivors[town_mask]
                
                # C. Filter Floor Area (on survivors)
                current_areas = db.columns['floor_area'][survivors]
                area_mask = current_areas >= y
                
                if not np.any(area_mask):
                    continue
                    
                survivors = survivors[area_mask]
                
                # Add to total matches
                matched_indices.extend(survivors)

            # --- AGGREGATION ---
            if not matched_indices:
                # No result logic
                # We do not write "No result" lines in CSV usually, or we skip?
                # The user prompt says "If there is no qualified data... please take 'No result' as query result"
                # But typical output file shows headers. We will handle this in output writing.
                continue

            matched_indices = np.array(matched_indices)
            
            # Calculate Price Per Sqm
            prices = db.columns['resale_price'][matched_indices]
            areas = db.columns['floor_area'][matched_indices]
            ppsm = prices / areas
            
            # Find Minimum
            min_idx_local = np.argmin(ppsm)
            best_global_idx = matched_indices[min_idx_local]
            best_ppsm = round(ppsm[min_idx_local])
            
            # Prepare Result Record
            # Need to decode the integers back to strings
            rec_month_str = db.decode_value('month', db.columns['month'][best_global_idx])
            rec_year = rec_month_str.split('-')[0]
            rec_mon = rec_month_str.split('-')[1]
            
            rec_town = db.decode_value('town', db.columns['town'][best_global_idx])
            rec_block = db.decode_value('block', db.columns['block'][best_global_idx]) # Block was encoded? No, Block is mixed. Check load.
            # In load, Block is encoded.
            
            rec_flat_model = db.decode_value('flat_model', db.columns['flat_model'][best_global_idx])
            rec_lease = db.columns['lease_commence_date'][best_global_idx]
            rec_area = db.columns['floor_area'][best_global_idx]
            
            # Format: (x, y),Year,Month,Town,Block,Floor_Area,Flat_Model,Lease_Commence_Date,Price_Per_Square_Meter
            res_row = [
                f"({x}, {y})", rec_year, rec_mon, rec_town, rec_block, 
                int(rec_area), rec_flat_model, rec_lease, int(best_ppsm)
            ]
            
            results.append(res_row)
            
            # Log for debug
            log_entries.append(f"x={x}, y={y}: Found {len(matched_indices)} matches. Min Price: {best_ppsm} at row {best_global_idx}")

    # --- SAVE OUTPUT ---
    out_filename = f"ScanResult_{matric_num}.csv"
    log_filename = f"ScanLog_{matric_num}.txt"
    
    # Headers as per demo
    csv_header = ["(x, y)","Year","Month","Town","Block","Floor_Area","Flat_Model","Lease_Commence_Date","Price_Per_Square_Meter"]
    
    with open(out_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        if not results:
            writer.writerow(["No result"])
        else:
            writer.writerows(results)
            
    with open(log_filename, 'w') as f:
        f.write("\n".join(log_entries))
        
    print(f"Finished. Saved {out_filename}")

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    # 1. Init and Load
    db = ColumnStoreDB()
    if os.path.exists(INPUT_FILE):
        db.load_csv(INPUT_FILE)
        
        # 2. Optimize
        db.optimize()
        
        # 3. User Interaction
        user_matric = input("Enter Matriculation Number (e.g., A6626226B): ").strip().upper()
        
        # 4. Run Query
        run_queries(db, user_matric)
        
        # 5. Clean up
        del db
        gc.collect()
        
    else:
        print(f"Error: {INPUT_FILE} not found.")