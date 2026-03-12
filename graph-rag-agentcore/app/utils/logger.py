# app/utils/logger.py

import logging
import os
from typing import Optional


_LOGGER_CACHE = {}


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Returns a configured logger.
    Safe for local usage and AWS Lambda.
    """

    logger_name = name or "graph-rag-agentcore"

    if logger_name in _LOGGER_CACHE:
        return _LOGGER_CACHE[logger_name]

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers (important in Lambda)
    if not logger.handlers:
        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    _LOGGER_CACHE[logger_name] = logger

    return logger
