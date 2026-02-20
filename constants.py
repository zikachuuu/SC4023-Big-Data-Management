import os

# Location of the input CSV file
# Currently set to be in the same directory as this constants.py file
INPUT_FILE = os.path.join(os.path.dirname(__file__), 'ResalePricesSingapore.csv')

# Directory to store log files
LOG_DIR = os.path.join(os.path.dirname(__file__), 'Logs')

# Directory to store csv output files
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'Results')

# Size for Zone Map chunks
# We compute the max and min of selected columns for each chunk and store them in the Zone Map
CHUNK_SIZE = 1000  

# Mapping from Matric Digit to Town (Source: Table 1)
TOWN_MAP_DIGIT = {
    0: "BEDOK", 1: "BUKIT PANJANG", 2: "CLEMENTI", 3: "CHOA CHU KANG", 4: "HOUGANG",
    5: "JURONG WEST", 6: "PASIR RIS", 7: "TAMPINES", 8: "WOODLANDS", 9: "YISHUN"
}

# Mapping from Month to Digit
MONTH_MAP_DIGIT = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}