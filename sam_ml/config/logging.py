import logging
import os
import sys


def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    
    log_level = str(os.getenv("SAM_ML_LOG_LEVEL")).lower()
    if log_level == "debug":
        logger.setLevel(logging.DEBUG)
    elif log_level == "info":
        logger.setLevel(logging.INFO)
    elif log_level == "warning":
        logger.setLevel(logging.WARNING)
    elif log_level == "error":
        logger.setLevel(logging.ERROR)
    else:
        os.environ["SAM_ML_LOG_LEVEL"] = "info"
        logger.setLevel(logging.INFO)

    c_handler = logging.StreamHandler(sys.stdout)

    c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)

    # to prevent double logging output
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.propagate = False

    logger.addHandler(c_handler)
    return logger
