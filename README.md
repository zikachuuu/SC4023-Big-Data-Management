# SC4023-Big-Data-Management

## Overview
Code for NTU SC4023's Semester Group Project, by Group 3

## Organisation of Repository
```
SC4023-Big-Data-Management/
├── Results/                                       # Folder containing the output CSV files
├── .gitignore                                     # Git ignore file
├── columnStoreDB.py                               # Columnar database class
├── constants.py                                   # Constants used in the project
├── generate_all_possible_matric_num.py            # Generates 100 test matriculation numbers
├── main.py                                        # Main script that executes the queries
├── README.md                                      # This file
├── requirements.txt                               # Python dependencies
├── ResalePricesSingapore.csv                      # Input CSV file for this project
├── utility.py                                     # Helper and auxillary functions
└── verify_scan_results.py                         # Verifies the correctness of the generated CSV files
```

## Setup
Begin by creating a virtual Python environment or Anaconda environment, and installing all required dependencies with the following command:
```bash
pip install -r requirements.txt
```

## Usage
### Step 1: Start the program
Run the main script with:
```bash
python main.py
```

### Step 2: Enter your matriculation number
The program will prompt you to enter your matriculation number. The input is case-insensitive, since the letters will not be used when parsing the matriculation number.

### Step 3: Generation of results
The program will log the execution process in the terminal and save the final results to a CSV file in the `Results` folder. A copy of the database state (`database_state.json`) and the log file (`run_<matriculation_num>.log`) will be saved in the `Logs` folder.

### Step 4: Verification of results
Verify the correctness of the generated CSV files with:
```bash
python verify_scan_results.py
```
If there are no mismatches, the program will print "ScanResult_XXXXXXXXX.csv: PASS (all 568 pairs matched)" where XXXXXXXXX is your matriculation number. If there are mismatches, the program will print "FAIL" and list the mismatched rows.

### Step 5 (optional): Rigorous verification of program
Generate 100 test matriculation numbers and their corresponding output CSV files with:
```bash
python generate_all_possible_matric_num.py
```

Then, verify the correctness of the generated CSV files with:
```bash
python verify_scan_results.py
```