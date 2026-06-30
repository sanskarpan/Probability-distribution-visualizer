"""Structured logging infrastructure with JSON and console output."""

import functools
import json
import logging
import sys
import threading
import time
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, TypeVar

_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
_loggers_initialized: Dict[str, bool] = {}
_setup_lock = threading.Lock()

LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

F = TypeVar("F", bound=Callable[..., Any])


class _StructuredFormatter(logging.Formatter):
    """JSON structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        corr_id = _correlation_id.get()
        if corr_id:
            log_entry["correlation_id"] = corr_id

        thread_name = threading.current_thread().name
        log_entry["thread"] = thread_name

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        extra_data = getattr(record, "_extra", None)
        if extra_data:
            log_entry["data"] = extra_data

        return json.dumps(log_entry, default=str)


class _ConsoleFormatter(logging.Formatter):
    """Colorized console log formatter."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"
    DIM = "\033[2m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        corr_id = _correlation_id.get()

        parts = [
            f"{self.DIM}{timestamp}{self.RESET}",
            f"{color}{record.levelname:<8}{self.RESET}",
            f"[{record.name}]",
        ]

        if corr_id:
            parts.append(f"[{corr_id[:8]}]")

        parts.append(record.getMessage())

        if record.exc_info and record.exc_info[1]:
            tb = "".join(traceback.format_exception(*record.exc_info))
            parts.append(f"\n{self.DIM}{tb}{self.RESET}")

        return " ".join(parts)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set or generate a correlation ID for request tracking."""
    cid = correlation_id or str(uuid.uuid4())
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return _correlation_id.get()


def setup_logger(
    name: str = "root",
    level: int = logging.INFO,
    structured: bool = False,
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.

    Args:
        name: Logger name (use __name__ for module-level loggers).
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        structured: If True, emit JSON lines; otherwise pretty console output.

    Returns:
        Configured logger instance.
    """
    with _setup_lock:
        if name in _loggers_initialized:
            return logging.getLogger(name)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        if logger.handlers:
            logger.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        if structured:
            handler.setFormatter(_StructuredFormatter())
        else:
            handler.setFormatter(_ConsoleFormatter())

        logger.addHandler(handler)
        _loggers_initialized[name] = True

        return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the default console configuration.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance (created with default setup if not yet initialized).
    """
    if name not in _loggers_initialized:
        return setup_logger(name)
    return logging.getLogger(name)


def log_error(
    logger: logging.Logger,
    message: str,
    exc: Optional[Exception] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an error with full exception context.

    Args:
        logger: Logger instance to use.
        message: Human-readable error message.
        exc: Exception instance (if available, to attach traceback).
        extra: Optional additional context data.
    """
    extra_data = extra or {}
    if exc:
        extra_data["error_type"] = type(exc).__name__
        extra_data["error_message"] = str(exc)

    record = logger.makeRecord(
        logger.name,
        logging.ERROR,
        "(unknown)",
        0,
        message,
        (),
        exc_info=sys.exc_info() if exc is None else (type(exc), exc, exc.__traceback__),
    )
    if extra_data:
        record._extra = extra_data  # type: ignore[attr-defined]
    logger.handle(record)


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.

    Usage:
        @log_execution_time(logger)
        def my_function():
            ...

    Args:
        logger: Logger instance to use. If None, logger for the module is used.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _logger = logger or get_logger(func.__module__)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                _logger.debug(
                    "%s.%s completed in %.4f ms",
                    func.__module__,
                    func.__qualname__,
                    elapsed * 1000,
                )
                return result
            except Exception:
                elapsed = time.perf_counter() - start
                _logger.warning(
                    "%s.%s failed after %.4f ms",
                    func.__module__,
                    func.__qualname__,
                    elapsed * 1000,
                )
                raise

        return wrapper  # type: ignore[return-value]

    if callable(logger):
        func = logger
        logger = None
        return decorator(func)

    return decorator