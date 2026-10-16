"""Centralized structured logging configuration.

Provides structured console/file logging, standard-library logging integration,
context-variable propagation, and optional Sentry error reporting.

Configure once during application startup, before creating application workers.
"""

import logging
import logging.handlers
import sys
import threading
from typing import Any

import sentry_sdk
import structlog
from sentry_sdk.integrations.logging import LoggingIntegration

from dialogue_system.config.settings import Environment, get_settings


class LoggingLifecycle:
    """Track the process-local state of logging configuration.

    This internal helper provides thread-safe state management for the
    application logging lifecycle. It prevents multiple concurrent callers
    from configuring the logging system more than once.
    """

    # Initialize logging configuration state variables
    _lock = threading.RLock()
    _is_configured = False
    _handlers: list[logging.Handler] = []  # noqa: RUF012
    _sentry_initialized = False

    @classmethod
    def is_configured(cls) -> bool:
        """Return whether application logging has been configured."""
        return cls._is_configured


def _get_shared_processors() -> list[Any]:
    """Return processors shared by structlog and stdlib logging.

    These processors ensure that logs coming from both structlog and
    standard logging.info/error calls share the same metadata (timestamp, level, etc).

    Returns
    -------
    list[Any]
        A list of processors to be used in structlog configuration.
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(
            fmt="iso",
            utc=True,
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]


def _make_formatter(*, json_output: bool) -> logging.Formatter:
    """Build a formatter for structured and standard-library logs.

    Parameters
    ----------
    json_output : bool
        Whether to output logs in JSON format (True) or human-readable
        console format (False).

    Returns
    -------
    logging.Formatter
        A configured logging formatter for use with logging handlers.
    """
    # Define the processors to be used in the formatter
    processors = [
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
    ]

    # Add the appropriate renderer based on the desired output format
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_get_shared_processors(),
        processors=processors,
    )


def _get_file_rotation_when(value: str) -> str:
    """Normalize a supported rotating-file interval.

    Parameters
    ----------
    value : str
        The user-provided interval for log file rotation.

    Returns
    -------
    str
        The normalized interval string, suitable for
        use with TimedRotatingFileHandler.
    """
    normalized = value.strip()

    if normalized.lower() == "midnight":
        return "midnight"

    return normalized.upper()


def _build_file_handler(settings: Any) -> logging.Handler:
    """Create a rotating JSON log file handler.

    Parameters
    ----------
    settings : Any
        The application settings object containing
        logging configuration.

    Returns
    -------
    logging.Handler
        A configured rotating file handler.
    """
    handler = logging.handlers.TimedRotatingFileHandler(
        filename=settings.LOGS_DIR / "application.log",
        when=_get_file_rotation_when(settings.LOG_ROTATION_WHEN),
        interval=1,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
        utc=True,
    )

    handler.setFormatter(_make_formatter(json_output=True))
    return handler


def _configure_sentry() -> None:
    """Initialize Sentry error monitoring for the application."""
    settings = get_settings()

    if LoggingLifecycle._sentry_initialized:
        return

    if not settings.SENTRY_DSN:
        return

    if settings.ENVIRONMENT not in {Environment.STAGING, Environment.PRODUCTION}:
        return

    sentry_sdk.init(
        dsn=str(settings.SENTRY_DSN),
        environment=settings.ENVIRONMENT.value,
        integrations=[
            LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR,
            )
        ],
        send_default_pii=False,
        attach_stacktrace=True,
        traces_sample_rate=0.0,
    )

    LoggingLifecycle._sentry_initialized = True


def configure_logging(*, force: bool = False) -> None:
    """Configure application-wide structured logging."""
    with LoggingLifecycle._lock:
        if LoggingLifecycle.is_configured() and not force:
            return

        settings = get_settings()
        is_production = settings.ENVIRONMENT in {Environment.STAGING, Environment.PRODUCTION}

        # Remove existing handlers to prevent duplicate logs
        for handler in LoggingLifecycle._handlers:
            root = logging.getLogger()
            root.removeHandler(handler)
            handler.close()

        LoggingLifecycle._handlers.clear()

        shared_processors = _get_shared_processors()

        # Structlog configuration
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(settings.LOG_LEVEL.value)

        # Standard output handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(_make_formatter(json_output=is_production))

        root_logger.addHandler(console_handler)
        LoggingLifecycle._handlers.append(console_handler)

        # Optional file logging for local development debugging
        if settings.ENVIRONMENT == Environment.DEVELOPMENT and settings.LOG_TO_FILE:
            file_handler = _build_file_handler(settings)
            root_logger.addHandler(file_handler)
            LoggingLifecycle._handlers.append(file_handler)

        # Silencing/Rerouting noisy third-party logs.
        for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "livekit"):
            third_party_logger = logging.getLogger(logger_name)
            third_party_logger.handlers.clear()
            third_party_logger.propagate = True

        # Capture Python warnings (e.g. DeprecationWarning) and send them to logs.
        logging.captureWarnings(capture=True)

        # Initialize Sentry error monitoring and tracing
        _configure_sentry()

        LoggingLifecycle._is_configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured application logger.

    Logging configuration is initialized lazily if it has not already been
    configured.
    """
    if not LoggingLifecycle.is_configured():
        configure_logging()

    settings = get_settings()
    return structlog.get_logger(name or settings.LOGGER_NAME)


if __name__ == "__main__":
    configure_logging()
    logger = get_logger(__name__)
    logger.info("Logging system initialized successfully.")
