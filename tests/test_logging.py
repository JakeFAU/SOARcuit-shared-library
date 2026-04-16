import io
import json
import logging
import sys

import structlog
from shared.domain.tracing import initialize_logger
from shared.logging import (
    LogFormat,
    bind_log_context,
    clear_log_context,
    configure_logging,
    get_logger,
)


def _reset_logging_state() -> None:
    clear_log_context()
    structlog.reset_defaults()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.NOTSET)


def test_configure_logging_emits_structured_json(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    _reset_logging_state()

    configure_logging(log_format=LogFormat.JSON, force=True)
    get_logger("tests.logging").info("hello", answer=42)

    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "hello"
    assert payload["answer"] == 42
    assert payload["logger"] == "tests.logging"
    assert payload["level"] == "info"
    assert "timestamp" in payload


def test_initialize_logger_uses_central_configuration(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    _reset_logging_state()

    configure_logging(log_format=LogFormat.JSON, force=True)
    bind_log_context(request_id="req-123")
    try:
        initialize_logger("service-a").info("context-test")
    finally:
        clear_log_context()

    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "context-test"
    assert payload["logger"] == "service-a"
    assert payload["request_id"] == "req-123"
