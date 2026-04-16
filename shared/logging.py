from __future__ import annotations

import logging
import os
import sys
from enum import StrEnum
from typing import Any

import structlog
from structlog.types import Processor
from structlog_gcp import build_processors as build_gcp_processors

DEFAULT_LOG_LEVEL = "INFO"
_AUTO_CONFIGURE_ENV_VAR = "SOAR_AUTO_CONFIGURE_LOGGING"
_FALSEY_VALUES = {"0", "false", "no", "off"}


class LogFormat(StrEnum):
    CONSOLE = "console"
    JSON = "json"
    GCP = "gcp"


def configure_logging(
    *,
    service_name: str | None = None,
    version: str | None = None,
    level: str | int | None = None,
    log_format: LogFormat | str | None = None,
    force: bool = False,
) -> None:
    """Configure structlog and the root stdlib logger for all importing apps.

    The default format is chosen from environment and runtime context:
    - `SOAR_LOG_FORMAT` / `LOG_FORMAT` when set
    - `gcp` when running in Google Cloud environments
    - `console` on an interactive terminal
    - `json` otherwise
    """

    if structlog.is_configured() and not force:
        return

    resolved_level = _coerce_log_level(level)
    resolved_format = _coerce_log_format(log_format)

    root_logger = logging.getLogger()
    if force or not root_logger.handlers:
        logging.basicConfig(
            level=resolved_level,
            format="%(message)s",
            stream=sys.stdout,
            force=force,
        )
    else:
        root_logger.setLevel(resolved_level)

    structlog.configure(
        processors=_build_processors(
            log_format=resolved_format,
            service_name=service_name,
            version=version,
        ),
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def auto_configure_logging() -> None:
    """Configure logging on import unless the caller opted out."""
    env_value = os.getenv(_AUTO_CONFIGURE_ENV_VAR, "true").strip().lower()
    if env_value in _FALSEY_VALUES:
        return
    configure_logging()


def get_logger(name: str | None = None, **bindings: Any) -> Any:
    """Return a configured structlog logger."""
    if not structlog.is_configured():
        configure_logging()

    logger = structlog.get_logger(name) if name else structlog.get_logger()
    return logger.bind(**bindings) if bindings else logger


def bind_log_context(**values: Any) -> None:
    """Bind values into the structlog contextvar store."""
    structlog.contextvars.bind_contextvars(**values)


def clear_log_context() -> None:
    """Clear any previously bound structlog contextvars."""
    structlog.contextvars.clear_contextvars()


def _build_processors(
    *,
    log_format: LogFormat,
    service_name: str | None,
    version: str | None,
) -> list[Processor]:
    if log_format == LogFormat.GCP:
        return [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            *build_gcp_processors(service_name, version),
        ]

    renderer = (
        structlog.dev.ConsoleRenderer()
        if log_format == LogFormat.CONSOLE
        else structlog.processors.JSONRenderer()
    )

    return [
        structlog.stdlib.filter_by_level,
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        renderer,
    ]


def _coerce_log_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level

    raw_level = level or os.getenv("SOAR_LOG_LEVEL") or os.getenv("LOG_LEVEL") or DEFAULT_LOG_LEVEL
    resolved = logging.getLevelName(str(raw_level).upper())
    if isinstance(resolved, int):
        return resolved
    raise ValueError(f"Unsupported log level: {raw_level}")


def _coerce_log_format(log_format: LogFormat | str | None) -> LogFormat:
    if isinstance(log_format, LogFormat):
        return log_format
    if isinstance(log_format, str):
        return LogFormat(log_format.strip().lower())

    env_format = os.getenv("SOAR_LOG_FORMAT") or os.getenv("LOG_FORMAT")
    if env_format:
        return LogFormat(env_format.strip().lower())

    if os.getenv("K_SERVICE") or os.getenv("K_REVISION") or os.getenv("GOOGLE_CLOUD_PROJECT"):
        return LogFormat.GCP

    return LogFormat.CONSOLE if sys.stdout.isatty() else LogFormat.JSON


auto_configure_logging()
