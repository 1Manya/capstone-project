"""
logger.py
---------
Minimal logger factory. train.py imports `get_logger` from here - this file
didn't exist yet, so training would ImportError without it.

Just a thin wrapper around Python's standard logging module: prints
timestamped messages to stdout, and guards against attaching duplicate
handlers if get_logger() is called more than once with the same name
(e.g. once per ensemble specialist).
"""

import logging
import sys


def get_logger(name: str = "app") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on repeated calls
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


# -- Quick test ----------------------------------------------------------------
if __name__ == "__main__":
    log = get_logger("test")
    log.info("logger OK")