"""Application Settings and Configuration Management.

This module defines the type-safe application configuration schema using
Pydantic Settings and Pydantic v2. It manages environment variables, `.env` file
overrides, runtime directories, and hyperparameter configuration (`params.yaml`).
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import (
    Field,
    HttpUrl,
    SecretStr,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root(current_path: Path) -> Path:
    """Locate the project root by searching for known anchor files.

    Parameters
    ----------
    current_path : pathlib.Path
        Starting path from which the parent directories are searched.

    Returns
    -------
    pathlib.Path
        Path to the detected project root directory.
    """
    for parent in [current_path, *list(current_path.parents)]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    return current_path.resolve().parents[2]


# Get the base directory for the project
BASE_DIR = _find_project_root(Path(__file__).resolve())


class Environment(StrEnum):
    """Enumeration of supported application logging levels."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(StrEnum):
    """Enumeration of logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Define validated application-wide runtime configuration.

    Configuration values are loaded from environment variables and the
    project-level ``.env`` file, with environment variables taking
    precedence.

    Notes
    -----
    Unknown configuration values are ignored, and environment variable
    names are matched case-insensitively.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Runtime configuration
    APP_NAME: str = Field(default="Dialogue System", min_length=1)
    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT)

    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO)
    LOGGER_NAME: str = Field(default="dialogue_system", min_length=1)
    LOG_TO_FILE: bool = Field(default=True)
    LOG_BACKUP_COUNT: int = Field(default=14, ge=1)
    LOG_ROTATION_WHEN: str = Field(default="midnight", pattern=r"^(midnight|[Hh]|[Dd]|[Ww])$")

    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000, ge=1, le=65535)
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    PUBLIC_FRONTEND_URL: str = Field(default="http://localhost:5173")

    # Workspace Paths
    LOGS_DIR: Path = Field(default_factory=lambda: BASE_DIR / "logs")
    WHISPER_DOWNLOAD_DIR: Path = Field(default_factory=lambda: BASE_DIR / "models" / "whisper")
    ARGOS_DOWNLOAD_DIR: Path = Field(default_factory=lambda: BASE_DIR / "models" / "argos")
    PIPER_DOWNLOAD_DIR: Path = Field(default_factory=lambda: BASE_DIR / "models" / "piper")

    # Sentry
    SENTRY_DSN: HttpUrl | None = Field(default=None)

    # LiveKit
    LIVEKIT_URL: str | None = Field(default=None)
    LIVEKIT_API_KEY: SecretStr | None = Field(default=None)
    LIVEKIT_API_SECRET: SecretStr | None = Field(default=None)
    LIVEKIT_AGENT_NAME: str = Field(default="dialogue-translator")

    # Whisper
    WHISPER_MODEL: str = Field(default="small")
    WHISPER_DEVICE: str = Field(default="cpu", pattern=r"^(cpu|cuda)$")
    WHISPER_COMPUTE_TYPE: str = Field(default="int8", pattern=r"^(float32|float16|int8)$")

    # Voice Activity Detection
    VAD_AGGRESSIVENESS: float = Field(default=0.45, ge=0, le=3)
    MAX_UTTERANCE_DURATION: float = Field(default=20.0, ge=1.0)
    TRANSLATION_ENABLED: bool = Field(default=True)
    TTS_ENABLED: bool = Field(default=True)

    # Model Setup Configuration
    MODEL_LANGUAGES: list[str] = Field(default_factory=lambda: ["en", "hi"])
    PIPER_VOICES_BASE_URL: str = Field(
        default="https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    )
    PIPER_VOICES: dict[str, str] = Field(
        default_factory=lambda: {
            "en_male": "en/en_US/mike/medium",
            "en_female": "en/en_US/kristin/medium",
            "hi_male": "hi/hi_IN/pratham/medium",
            "hi_female": "hi/hi_IN/priyamvada/medium",
        }
    )

    @model_validator(mode="after")
    def _validate_environment_credentials(self) -> Self:
        """Validate credentials required by the configured runtime environment.

        Raises
        ------
        ValueError
            If a required credential is missing or invalid for the selected
            environment.
        """
        if self.ENVIRONMENT not in {
            Environment.PRODUCTION,
            Environment.STAGING,
        }:
            return self

        missing_secrets: list[str] = []

        if self.SENTRY_DSN is None:
            missing_secrets.append("SENTRY_DSN")

        if self.LIVEKIT_URL is None:
            missing_secrets.append("LIVEKIT_URL")

        if self.LIVEKIT_API_KEY is None:
            missing_secrets.append("LIVEKIT_API_KEY")

        if self.LIVEKIT_API_SECRET is None:
            missing_secrets.append("LIVEKIT_API_SECRET")

        if missing_secrets:
            raise ValueError(
                f"Missing required secrets for "
                f"{self.ENVIRONMENT.value} environment: "
                f"{', '.join(missing_secrets)}"
            )

        return self

    def ensure_directories(self) -> None:
        """Create required application workspace directories."""
        directories = [
            self.LOGS_DIR,
            self.WHISPER_DOWNLOAD_DIR,
            self.ARGOS_DOWNLOAD_DIR,
            self.PIPER_DOWNLOAD_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings.

    Local development may use a project-level environment file, while
    deployment environments such as Render provide configuration directly
    through environment variables.
    """
    settings = Settings()
    settings.ensure_directories()

    return settings
