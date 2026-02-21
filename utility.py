import logging
import os

from constants import LOG_DIR, MONTH_MAP_DIGIT


def configure_logging(matriculation_number: str) -> logging.Logger:
    """
    Only log INFO and above to console, but log DEBUG and above to file.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    log_path = os.path.join(LOG_DIR, f"run_{matriculation_number}.log")

    logger = logging.getLogger("column_store_db")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. File: {log_path}")
    return logger


def convert_str_date_to_code(date_str: str) -> int:
    """
    Convert a date string in the format "MMM-YY" (e.g. "Jan-20") to an integer representation.
    We can use this integer representation for efficient storage and comparison in the column store database.

    The conversion is done by encoding the month and year into a single integer using the formula:
        encoded_date = int ((2 digit year) + (2 digit month)), where + is string concatenation, not addition.
    For example:
        "Jan-20" -> month = 1, year = 20 -> encoded_date = 2001
        "Feb-20" -> month = 2, year = 20 -> encoded_date = 2002
    """
    month_str, year_str = date_str.split('-')
    month = MONTH_MAP_DIGIT[month_str]  # Convert month name to its corresponding digit
    year = int(year_str)

    # Encode the date as an integer using string concatenation
    encoded_date = int(f"{year:02d}{month:02d}")  # Format year and month as 2-digit numbers and concatenate them
    return encoded_date

def convert_floor_area_to_code (floor_area: float) -> int:
    """
    Convert the floor area from a float to an integer code by multiplying by 10.
    This allows us to preserve one decimal place of precision while storing the value as an integer.
    For example:
        45.5 sqm -> 455
        30.0 sqm -> 300
    """
    return int(floor_area * 10)