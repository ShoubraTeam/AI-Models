"""
Logging setup
=============
Call setup_logging() once at application entry point.
"""

import logging
import sys
from pathlib import Path

from config.settings import LOG_LEVEL, LOG_FILE


def setup_logging(log_file: str | Path = LOG_FILE, level: str = LOG_LEVEL) -> None:
    """Configure root logger with console + rotating file handlers."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)-35s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # File
    try:
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except PermissionError:
        root.warning("Could not open log file %s — logging to console only.", log_file)
