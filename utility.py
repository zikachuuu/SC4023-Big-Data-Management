import logging
import os
from datetime import datetime


def configure_logging(matriculation_number: str) -> logging.Logger:
    """
    Only log INFO and above to console, but log DEBUG and above to file.
    """
    logs_dir = os.path.join(os.path.dirname(__file__), "Logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"run_{matriculation_number}_{timestamp}.log")

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
