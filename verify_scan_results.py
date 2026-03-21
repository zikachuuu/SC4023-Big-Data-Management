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
    # Year logic
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
    sql = f'''
    SELECT month, town, flat_type, block, street_name, storey_range, floor_area_sqm, flat_model, lease_commence_date, resale_price
    FROM (
        SELECT *, resale_price / floor_area_sqm AS psm
        FROM resale
        WHERE month IN ({months_str})
          AND town IN ({towns_str})
          AND floor_area_sqm >= {y}
          AND resale_price / floor_area_sqm <= 4725
    )
    WHERE psm = (
        SELECT MIN(resale_price / floor_area_sqm)
        FROM resale
        WHERE month IN ({months_str})
          AND town IN ({towns_str})
          AND floor_area_sqm >= {y}
          AND resale_price / floor_area_sqm <= 4725
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
    '''
    return sql

# Extract parameters from ScanResult filename (assumes ScanResult_MATRIC_X_Y.csv)
def extract_params_from_filename(filename):
    m = re.match(r"ScanResult_([A-Z0-9]+)_([0-9]+)_([0-9]+)\.csv", filename)
    if not m:
        return None
    matric, x, y = m.group(1), int(m.group(2)), int(m.group(3))
    return matric, x, y

# Path to the directory containing ScanResult files
directory = './Results/'
# Path to the CSV data file
csv_file = './ResalePricesSingapore.csv'
# Path to the SQLite database
sqlite_db = 'resale_prices.db'

# Load CSV into SQLite (if not already loaded)
def load_csv_to_sqlite():
    conn = sqlite3.connect(sqlite_db)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS resale (
        month TEXT, town TEXT, flat_type TEXT, block TEXT, street_name TEXT, storey_range TEXT,
        floor_area_sqm REAL, flat_model TEXT, lease_commence_date INTEGER, resale_price INTEGER
    )''')
    conn.commit()
    # Check if table is empty
    c.execute('SELECT COUNT(*) FROM resale')
    if c.fetchone()[0] == 0:
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                c.execute('INSERT INTO resale VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', row)
        conn.commit()
    conn.close()

# Compare ScanResult file to SQLite query result

# Compare a single (x, y) result row in ScanResult to SQLite query result
def compare_single_xy(scan_row, db_rows):
    # db_rows: list of rows (each is a list of str)
    # scan_row: a list of str (from ScanResult)
    return scan_row in db_rows

if __name__ == '__main__':
    load_csv_to_sqlite()
    for filename in os.listdir(directory):
        if filename.startswith('ScanResult') and filename.endswith('.csv'):
            m = re.match(r"ScanResult_([A-Z0-9]+)\.csv", filename)
            if not m:
                print(f"Skipping {filename}: could not parse matric number.")
                continue
            matric = m.group(1)
            start_year, start_month, towns = parse_matriculation(matric)
            scanresult_path = os.path.join(directory, filename)
            # Read ScanResult file into dict keyed by (x, y)
            scan_dict = {}
            with open(scanresult_path, newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if row and row[0].startswith('('):
                        xy = row[0].strip('()').split(',')
                        if len(xy) == 2:
                            xval, yval = int(xy[0]), int(xy[1])
                            scan_dict[(xval, yval)] = row[1:]  # exclude (x,y) column
            total = 0
            mismatches = 0
            mismatch_details = []
            for x in range(1, 9):
                for y in range(80, 151):
                    query = generate_sql_query(start_year, start_month, x, y, towns)
                    conn = sqlite3.connect(sqlite_db)
                    c = conn.cursor()
                    c.execute(query)
                    db_rows = [list(map(str, row)) for row in c.fetchall()]
                    conn.close()
                    if db_rows:
                        db_row = db_rows[0]
                        monthstr = db_row[0]
                        mth, yr = monthstr.split('-')
                        year_out = f"20{yr}"
                        month_out = f"{MONTH_NUM_MAP[mth]:02d}"
                        town = db_row[1]
                        block = db_row[3]
                        floor_area = str(float(db_row[6]))
                        flat_model = db_row[7]
                        lease_commence = db_row[8]
                        price_per_sqm = str(int(float(db_row[9]) / float(db_row[6])))
                        expected = [year_out, month_out, town, block, floor_area, flat_model, lease_commence, price_per_sqm]
                    else:
                        expected = ["No Result"] * 8
                    scan_row = scan_dict.get((x, y), None)
                    if scan_row is None or scan_row != expected:
                        mismatches += 1
                        mismatch_details.append({
                            'x': x,
                            'y': y,
                            'expected': expected,
                            'actual': scan_row
                        })
                    total += 1
            if mismatches == 0:
                print(f"{filename}: PASS (all {total} (x,y) pairs matched)")
            else:
                print(f"{filename}: FAIL ({mismatches} mismatches out of {total} (x,y) pairs)")
                for detail in mismatch_details:
                    print(f"  MISMATCH (x={detail['x']}, y={detail['y']}):\n    Expected: {detail['expected']}\n    Actual:   {detail['actual']}")