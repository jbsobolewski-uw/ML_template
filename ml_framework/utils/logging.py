"""
utils/logging.py
----------------
Centralised logging configuration for the framework.
Call setup_logging() once in main.py before spawning workers.
Workers inherit the log level but configure their own handlers
to avoid cross-process handler conflicts.
"""

import logging
import sys


_FMT = "[%(asctime)s] [%(levelname)s] [%(processName)s/%(name)s] %(message)s"
_DATE_FMT = "%H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger with a StreamHandler to stdout.

    Parameters
    ----------
    level : str
        Logging level name ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric)


def get_worker_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Return a logger for use inside a spawned worker process.
    Configures the child process's root logger so that ALL modules (accelerator,
    neural, etc.) can successfully output logs. Uses sys.__stdout__ to bypass
    stream interception or buffering by multiprocessing.Pool.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module).
    level : str
        Logging level name.
    """
    import sys

    # Force low-level un-redirected console streams to use line buffering
    if hasattr(sys.__stdout__, "reconfigure"):
        sys.__stdout__.reconfigure(line_buffering=True)
    if hasattr(sys.__stderr__, "reconfigure"):
        sys.__stderr__.reconfigure(line_buffering=True)

    numeric = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric)

    # Prevent duplicate handlers if get_worker_logger is called multiple times
    # in the same worker process lifetime
    if not root.handlers:
        handler = logging.StreamHandler(sys.__stdout__)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
        root.addHandler(handler)

    logger = logging.getLogger(name)
    logger.setLevel(numeric)

    return logger