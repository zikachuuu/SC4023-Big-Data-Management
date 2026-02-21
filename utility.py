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

    file_handler = logging.FileHandler(log_path, mode='w')  # 'w' to overwrite the log file for each run
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. File: {log_path}")
    return logger


def convert_month_str_to_code(month_str: str) -> int:
    """
    Convert a month string in the format "MMM-YY" (e.g. "Jan-20") to an integer representation.
    We can use this integer representation for efficient storage and comparison in the column store database.

    The conversion is done by encoding the month and year into a single integer using the formula:
        encoded_month = int ((2 digit year) + (2 digit month)), where + is string concatenation, not addition.
    For example:
        "Jan-20" -> month = 1, year = 20 -> encoded_month = 2001
        "Feb-20" -> month = 2, year = 20 -> encoded_month = 2002
    """
    month_str, year_str = month_str.split('-')
    month = MONTH_MAP_DIGIT[month_str]  # Convert month name to its corresponding digit
    year = int(year_str)

    # Encode the month as an integer using string concatenation
    encoded_month = int(f"{year:02d}{month:02d}")  # Format year and month as 2-digit numbers and concatenate them
    return encoded_month


def convert_code_to_month_str(code: int) -> str:
    """
    Convert an integer code back to a month string in the format "MMM-YY".
    This is the inverse of the convert_month_str_to_code function.

    The conversion is done by decoding the integer code back into month and year using string slicing.
    For example:
        2001 -> year = 20, month = 1 -> "Jan-20"
        2002 -> year = 20, month = 2 -> "Feb-20"
    """
    code_str = f"{code:04d}"  # Ensure the code is treated as a 4-digit number with leading zeros if necessary
    year_str = code_str[:2]   # First two digits represent the year
    month_str = code_str[2:]  # Last two digits represent the month

    # Convert month digit back to month name
    month_digit = int(month_str)
    month_name = next(key for key, value in MONTH_MAP_DIGIT.items() if value == month_digit)

    return f"{month_name}-{year_str}"


def convert_month_year_to_code(month: int, year: int) -> int:
    """
    Convert a month and year to the same integer code format as convert_month_str_to_code.
    This is useful for converting the target month and year from the query into the same format as the encoded month column in the database, so that we can perform comparisons using the encoded integer values.

    The conversion is done by encoding the month and year into a single integer using the formula:
        encoded_month = int ((2 digit year) + (2 digit month)), where + is string concatenation, not addition.
    For example:
        month = 1, year = 20 -> encoded_month = 2001
        month = 2, year = 20 -> encoded_month = 2002
    """
    encoded_month = int(f"{year:02d}{month:02d}")  # Format year and month as 2-digit numbers and concatenate them
    return encoded_month


def convert_floor_area_to_code (floor_area: float) -> int:
    """
    Convert the floor area from a float to an integer code by multiplying by 10.
    This allows us to preserve one decimal place of precision while storing the value as an integer.
    For example:
        45.5 sqm -> 455
        30.0 sqm -> 300
    """
    return int(floor_area * 10)