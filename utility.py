import logging
import os

from constants import LOG_DIR


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
