"""Tests for structured logging infrastructure."""

import io
import logging
import sys
import os
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.logger import (
    get_correlation_id,
    set_correlation_id,
    setup_logger,
    get_logger,
    log_error,
    log_execution_time,
)
import utils.logger as logger_mod


def _capture_handler_output(logger, fn):
    stream = io.StringIO()
    old_handlers = logger.handlers[:]
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    try:
        fn()
    finally:
        logger.handlers.clear()
        for h in old_handlers:
            logger.addHandler(h)
    return stream.getvalue()


class TestCorrelationId:
    def test_set_and_get(self):
        cid = set_correlation_id()
        assert isinstance(cid, str)
        assert uuid.UUID(cid)
        assert get_correlation_id() == cid

    def test_set_custom(self):
        cid = set_correlation_id("my-custom-id")
        assert cid == "my-custom-id"
        assert get_correlation_id() == "my-custom-id"

    def test_set_none_generates_uuid(self):
        cid = set_correlation_id(None)
        assert isinstance(cid, str)
        assert len(cid) == 36


class TestSetupLogger:
    def teardown_method(self):
        import src.utils.logger as logger_mod
        logger_mod._loggers_initialized.clear()

    def test_setup_console(self):
        logger = setup_logger("test_console", level=logging.DEBUG)
        assert logger.level == logging.DEBUG
        assert logger.propagate is False
        assert len(logger.handlers) == 1

    def test_setup_structured(self):
        logger = setup_logger("test_json", level=logging.WARNING, structured=True)
        assert logger.level == logging.WARNING
        assert len(logger.handlers) == 1

    def test_setup_idempotent(self):
        logger1 = setup_logger("test_idem", level=logging.INFO)
        logger2 = setup_logger("test_idem", level=logging.DEBUG)
        assert logger1 is logger2
        assert logger2.level == logging.INFO

    def test_get_logger_creates(self):
        import src.utils.logger as logger_mod
        logger_mod._loggers_initialized.clear()
        logger = get_logger("test_get_create")
        assert logger.propagate is False

    def test_get_logger_cached(self):
        logger1 = get_logger("test_cached")
        logger2 = get_logger("test_cached")
        assert logger1 is logger2


class TestLogError:
    def test_log_error_no_exc(self):
        logger = setup_logger("test_log_err")
        output = _capture_handler_output(
            logger, lambda: log_error(logger, "something went wrong")
        )
        assert "something went wrong" in output

    def test_log_error_with_exception(self):
        logger = setup_logger("test_log_exc")
        exc = ValueError("bad value")
        output = _capture_handler_output(
            logger,
            lambda: log_error(logger, "value error occurred", exc=exc, extra={"key": "val"}),
        )
        assert "value error occurred" in output
        assert "bad value" in output

    def test_log_error_with_extra(self):
        logger = setup_logger("test_log_extra")
        output = _capture_handler_output(
            logger, lambda: log_error(logger, "extra test", extra={"user_id": 42})
        )
        assert "extra test" in output


class TestLogExecutionTime:
    def test_success_logs_timing(self):
        logger = setup_logger("test_timer_success", level=logging.DEBUG)

        @log_execution_time(logger)
        def fast_func():
            return "ok"

        output = _capture_handler_output(logger, fast_func)
        assert "completed in" in output

    def test_failure_logs_warning(self):
        logger = setup_logger("test_timer_fail", level=logging.WARNING)

        @log_execution_time(logger)
        def broken_func():
            raise RuntimeError("boom")

        output = _capture_handler_output(
            logger, lambda: exec("try:\n broken_func()\nexcept RuntimeError:\n pass", {"broken_func": broken_func})
        )
        assert "failed after" in output

    def test_with_explicit_logger(self):
        logger = setup_logger("custom_timer", level=logging.DEBUG)

        @log_execution_time(logger)
        def slow_func():
            return 42

        output = _capture_handler_output(logger, slow_func)
        assert "completed in" in output

    def test_direct_call_no_args(self):
        @log_execution_time
        def direct_func():
            return "x"

        logger = get_logger(direct_func.__module__)
        logger.setLevel(logging.DEBUG)
        output = _capture_handler_output(logger, direct_func)
        assert "completed in" in output


class TestStructuredFormatter:
    def test_formats_json(self):
        fmt = logger_mod._StructuredFormatter()
        record = logging.LogRecord(
            "test", logging.INFO, "path.py", 10, "hello world", (), None
        )
        output = fmt.format(record)
        import json
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello world"
        assert "timestamp" in parsed

    def test_formatter_includes_corr_id(self):
        set_correlation_id("abc-123-xyz")
        fmt = logger_mod._StructuredFormatter()
        record = logging.LogRecord("test", logging.INFO, "path.py", 10, "msg", (), None)
        output = fmt.format(record)
        import json
        parsed = json.loads(output)
        assert parsed["correlation_id"] == "abc-123-xyz"


class TestConsoleFormatter:
    def test_console_formatter(self):
        fmt = logger_mod._ConsoleFormatter()
        record = logging.LogRecord("test", logging.WARNING, "m.py", 5, "watch out", (), None)
        output = fmt.format(record)
        assert "WARNING" in output
        assert "watch out" in output
