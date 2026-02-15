import os

# Location of the input CSV file
# Currently set to be in the same directory as this constants.py file
INPUT_FILE = os.path.join(os.path.dirname(__file__), 'ResalePricesSingapore.csv')

# Size for Zone Map chunks
# We compute the max and min of selected columns for each chunk and store them in the Zone Map
CHUNK_SIZE = 1000  

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