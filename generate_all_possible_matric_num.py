import os
import gc
from constants import INPUT_FILE, RESULTS_DIR
from columnStoreDB import ColumnStoreDB
from utility import configure_logging
from main import parse_matriculation, run_queries

def generate_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] {INPUT_FILE} not found.")
        return

    # 1. Initialize and load the database
    print("Initializing Database...")
    db = ColumnStoreDB()
    print("Loading CSV into Column Store Database...")
    db.load_csv(INPUT_FILE)
    print("Database loaded successfully.")

    # 2. Iterate through all 100 possible matriculation numbers
    for i in range(100):
        last_two = f'{i:02d}'
        matric_num = f'01234{last_two}C'
        
        print(f'Generating ScanResult for {matric_num}...')
        
        # Remove old ScanResult file if it exists
        scanresult_path = os.path.join(RESULTS_DIR, f'ScanResult_{matric_num}.csv')
        if os.path.exists(scanresult_path):
            os.remove(scanresult_path)

        # Configure logger for this specific run
        logger = configure_logging(matric_num)
        
        # Parse the target criteria based on the matriculation number
        (start_month_code, town_names) = parse_matriculation(matric_num)
        
        # 3. Run queries using the ALREADY LOADED database
        try:
            run_queries(db, start_month_code, town_names, matric_num, logger)
        except Exception as e:
            print(f'  [ERROR] Processing failed for {matric_num}:\n{e}')

    # Optional memory cleanup
    del db
    gc.collect()
    print('All ScanResult files generated.')

if __name__ == "__main__":
    generate_all()