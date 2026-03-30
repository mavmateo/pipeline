"""
logger.py
---------
Centralised logging. All modules call `get_logger(__name__)`.
Outputs structured logs to console + rotating file handler.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import get_config


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger configured once per process.
    Thread-safe: Python's logging module handles concurrent writes.
    """
    cfg = get_config().logging
    logger = logging.getLogger(name)

    # Guard: don't add handlers twice (happens on re-imports / tests)
    if logger.handlers:
        return logger

    logger.setLevel(cfg.level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Rotating file handler
    if cfg.log_to_file:
        log_path = Path(cfg.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path,
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Prevent log records bubbling up to the root logger
    logger.propagate = False
    return logger
