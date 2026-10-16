"""Structured Application Logging Module.

Implements a centralized logging system that provides formatted
console output and rotating file logs. It is designed to be thread-safe
and prevents duplicate handler initialization in concurrent environments.
"""

import logging
import logging.handlers
import sys
import threading
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import sentry_sdk
import structlog
from sentry_sdk.integrations.logging import LoggingIntegration

from dialogue_system.config.settings import get_settings


class LoggingLifecycle:
    """Track the process-local state of logging configuration.

    This internal helper provides thread-safe state management for the
    application logging lifecycle. It prevents multiple concurrent callers
    from configuring the logging system more than once.
    """

    _lock = threading.Lock()
    _is_configured = False

    @classmethod
    def is_configured(cls) -> bool:
        """Return whether application logging has been configured."""
        return cls._is_configured

    @classmethod
    def mark_configured(cls) -> None:
        """Mark application logging as successfully configured."""
        cls._is_configured = True


def _init_sentry(dsn: str, environment: str) -> None:
    """Initialize Sentry error monitoring for the application."""
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        integrations=[
            LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR,
            )
        ],
        attach_stacktrace=True,
        send_default_pii=False,
        traces_sample_rate=0.0,
    )


def _add_callsite_to_message(
    logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add source filename and line number to a log event."""
    _, _ = logger, method_name
    filename = event_dict.pop("filename", "unknown")
    lineno = event_dict.pop("lineno", "0")

    event = event_dict.get("event", "")
    event_dict["event"] = f"{filename}:{lineno} - {event}"
    return event_dict


def _get_shared_processors() -> list[Any]:
    """Create the shared Structlog processing pipeline."""
    return [
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
        _add_callsite_to_message,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]


def configure_logging(*, force: bool = False) -> None:
    """Configure application-wide structured logging."""
    if LoggingLifecycle.is_configured() and not force:
        return

    with LoggingLifecycle._lock:
        if LoggingLifecycle.is_configured() and not force:
            return

        settings = get_settings()
        is_sentry_environment = settings.ENVIRONMENT.value in {"STAGING", "PRODUCTION"}

        if settings.SENTRY_DSN and is_sentry_environment:
            _init_sentry(
                str(settings.SENTRY_DSN),
                settings.ENVIRONMENT.value,
            )

        shared_processors = _get_shared_processors()

        structlog.configure(
            processors=[
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # MODIFIED: Disabled bracket_logging_level to fix the fragmented layout
        if is_sentry_environment:
            console_renderer = structlog.processors.JSONRenderer()
        else:
            console_renderer = structlog.dev.ConsoleRenderer(colors=True)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    console_renderer,
                ],
            )
        )

        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(
                settings.LOGS_DIR
                / (f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:4]}" + ".log")
            ),
            when=settings.LOG_ROTATION_WHEN,
            interval=1,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
            )
        )

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(settings.LOG_LEVEL.value.upper())

        for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
            third_party_logger = logging.getLogger(logger_name)
            third_party_logger.handlers.clear()
            third_party_logger.propagate = True

        logging.captureWarnings(capture=True)
        LoggingLifecycle.mark_configured()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured application logger."""
    if not LoggingLifecycle.is_configured():
        configure_logging()

    settings = get_settings()
    logger_name = name or settings.LOGGER_NAME
    return structlog.get_logger(logger_name)
