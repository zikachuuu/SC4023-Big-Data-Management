import re
import sqlite3
import csv
import os

# Town digit mapping (from Table 1)
TOWN_MAP_DIGIT = {
    0: "BEDOK", 1: "BUKIT PANJANG", 2: "CLEMENTI", 3: "CHOA CHU KANG", 4: "HOUGANG",
    5: "JURONG WEST", 6: "PASIR RIS", 7: "TAMPINES", 8: "WOODLANDS", 9: "YISHUN"
}

# Month digit to name
MONTH_MAP = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
# For formatting output
MONTH_NUM_MAP = {v: k for k, v in MONTH_MAP.items()}

def parse_matriculation(matric_num):
    digits = [int(d) for d in matric_num if d.isdigit()]
    last_digit = digits[-1]
    if last_digit >= 5:
        start_year = 10 + last_digit
    else:
        start_year = 20 + last_digit
    sec_last_digit = digits[-2]
    start_month = sec_last_digit if sec_last_digit != 0 else 10
    towns = [TOWN_MAP_DIGIT[d] for d in set(digits)]
    return start_year, start_month, towns

def generate_sql_query(start_year, start_month, x, y, towns):
    months = []
    year = start_year
    month = start_month
    for i in range(x):
        if month > 12:
            month = 1
            year += 1
        months.append(f"{MONTH_MAP[month]}-{year:02d}")
        month += 1
    
    months_str = ','.join([f"'{m}'" for m in months])
    towns_str = ','.join([f"'{t}'" for t in towns])
    
    # Query optimized to use pre-calculated psm column
    sql = f'''
    SELECT month, town, flat_type, block, street_name, storey_range, floor_area_sqm, flat_model, lease_commence_date, resale_price, psm
    FROM resale
    WHERE month IN ({months_str})
      AND town IN ({towns_str})
      AND floor_area_sqm >= {y}
      AND psm <= 4725
      AND psm = (
          SELECT MIN(psm)
          FROM resale
          WHERE month IN ({months_str})
            AND town IN ({towns_str})
            AND floor_area_sqm >= {y}
            AND psm <= 4725
      )
    ORDER BY 
        SUBSTR(month, 5, 2), 
        CASE SUBSTR(month, 1, 3) 
            WHEN 'Jan' THEN '01' WHEN 'Feb' THEN '02' WHEN 'Mar' THEN '03' 
            WHEN 'Apr' THEN '04' WHEN 'May' THEN '05' WHEN 'Jun' THEN '06' 
            WHEN 'Jul' THEN '07' WHEN 'Aug' THEN '08' WHEN 'Sep' THEN '09' 
            WHEN 'Oct' THEN '10' WHEN 'Nov' THEN '11' WHEN 'Dec' THEN '12' 
        END,
        town, block
    LIMIT 1
    '''
    return sql

# Path to the directory containing ScanResult files
directory = './Results/'
csv_file = './ResalePricesSingapore.csv'
sqlite_db = 'resale_prices.db'

def load_csv_to_sqlite():
    conn = sqlite3.connect(sqlite_db)
    c = conn.cursor()
    # Added 'psm' column to pre-calculate price per square meter
    c.execute('''CREATE TABLE IF NOT EXISTS resale (
        month TEXT, town TEXT, flat_type TEXT, block TEXT, street_name TEXT, storey_range TEXT,
        floor_area_sqm REAL, flat_model TEXT, lease_commence_date INTEGER, resale_price INTEGER,
        psm REAL
    )''')
    
    # Check if table is empty
    c.execute('SELECT COUNT(*) FROM resale')
    if c.fetchone()[0] == 0:
        print("Loading CSV to SQLite and building indexes...")
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            
            rows_to_insert = []
            for row in reader:
                # Calculate psm on load: resale_price / floor_area_sqm
                psm = float(row[9]) / float(row[6])
                row.append(psm)
                rows_to_insert.append(row)
                
            c.executemany('INSERT INTO resale VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', rows_to_insert)
            
        # Create indexes to massively speed up our specific queries
        c.execute('CREATE INDEX idx_month_town_area_psm ON resale(month, town, floor_area_sqm, psm)')
        conn.commit()
    conn.close()

if __name__ == '__main__':
    load_csv_to_sqlite()
    
    # Open DB connection ONCE globally, not 568 times inside the loops
    conn = sqlite3.connect(sqlite_db)
    c = conn.cursor()

    for filename in os.listdir(directory):
        if filename.startswith('ScanResult') and filename.endswith('.csv'):
            m = re.match(r"ScanResult_([A-Z0-9]+)\.csv", filename)
            if not m:
                continue
                
            matric = m.group(1)
            start_year, start_month, towns = parse_matriculation(matric)
            scanresult_path = os.path.join(directory, filename)
            
            scan_dict = {}
            with open(scanresult_path, newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if row and row[0].startswith('('):
                        xy = row[0].strip('()').split(',')
                        if len(xy) == 2:
                            scan_dict[(int(xy[0]), int(xy[1]))] = row[1:]
                            
            total = 0
            mismatches = 0
            mismatch_details = []
            
            for x in range(1, 9):
                for y in range(80, 151):
                    query = generate_sql_query(start_year, start_month, x, y, towns)
                    
                    # Execute query using the persistent connection
                    c.execute(query)
                    db_row = c.fetchone()
                    
                    if db_row:
                        db_row = list(map(str, db_row))
                        monthstr = db_row[0]
                        mth, yr = monthstr.split('-')
                        year_out = f"20{yr}"
                        month_out = f"{MONTH_NUM_MAP[mth]:02d}"
                        town = db_row[1]
                        block = db_row[3]
                        floor_area = str(float(db_row[6]))
                        flat_model = db_row[7]
                        lease_commence = db_row[8]
                        # Use the pre-calculated psm
                        price_per_sqm = str(int(float(db_row[10])))
                        
                        expected = [year_out, month_out, town, block, floor_area, flat_model, lease_commence, price_per_sqm]
                    else:
                        expected = ["No Result"] * 8
                        
                    scan_row = scan_dict.get((x, y), None)
                    if scan_row is None or scan_row != expected:
                        mismatches += 1
                        mismatch_details.append({'x': x, 'y': y, 'expected': expected, 'actual': scan_row})
                    total += 1
                    
            if mismatches == 0:
                print(f"{filename}: PASS (all {total} pairs matched)")
            else:
                print(f"{filename}: FAIL ({mismatches} mismatches out of {total} pairs)")
                for detail in mismatch_details:
                    print(f"  MISMATCH (x={detail['x']}, y={detail['y']}):\n    Expected: {detail['expected']}\n    Actual:   {detail['actual']}")

    conn.close()