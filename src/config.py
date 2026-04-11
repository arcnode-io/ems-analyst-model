import enum
import logging
import os
from datetime import UTC, datetime

from pydantic import BaseModel
import yaml


class LogLevel(enum.StrEnum):
    """Logging levels for the application."""

    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"
    DEBUG = "DEBUG"


class Config(BaseModel):  # noqa: D101
    log_level: LogLevel
    db_host: str
    db_name: str
    db_user: str
    db_port: int
    api_url: str
    prometheus_pushgateway: str
    mlflow_tracking_uri: str
    mlflow_model_name: str
    schedule_time: str
    prophet_daily_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    prophet_yearly_seasonality: bool = True


class _ConfigMap(BaseModel):
    local: Config
    beta: Config


def load_config() -> Config:
    """Load configuration from cfg.yml file based on environment.

    Reads the configuration file and returns the appropriate config
    based on the ENV environment variable (defaults to 'local').

    Returns:
        Config object for the current environment

    Raises:
        FileNotFoundError: If cfg.yml file is not found
        yaml.YAMLError: If cfg.yml contains invalid YAML

    Example:
        >>> config = load_config()  # Uses ENV=local by default
        >>> config.log_level
        <Level.INFO: 'INFO'>

    """
    with open("cfg.yml") as file:
        config_map = _ConfigMap(**yaml.safe_load(file))
        environment = os.environ.get("ENV", "local")
        match environment:
            case "beta":
                return config_map.beta
            case _:
                return config_map.local


class _ZuluFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        """Format log records with Zulu time timestamps.

        Args:
            record: The log record to format

        Returns:
            Formatted log string with Zulu timestamp, level, and message

        Example:
            >>> formatter = ZuluFormatter()
            >>> record = logging.LogRecord(...)
            >>> formatter.format(record)
            '2024-01-01T12:00:00.123456Z  INFO: Sample message'

        """
        zulu_time = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return f"{zulu_time}  {record.levelname}: {record.getMessage()}"


def setup_logger(config: Config) -> None:
    """Set up the application logger with colored output and Zulu time formatting.

    Configures logging with color-coded level names and custom Zulu timestamp
    formatting using the provided configuration.

    Args:
        config: Configuration object containing log level settings

    Example:
        >>> cfg = Config(log_level=Level.INFO)
        >>> setup_logger(cfg)
        >>> logging.info("Test message")  # Outputs with colors and Zulu time

    """
    # adds color
    for ind, lvl in enumerate(
        [logging.ERROR, logging.INFO, logging.WARNING, logging.DEBUG],
    ):
        logging.addLevelName(
            lvl,
            f"\033[0;3{ind + 1}m%s\033[1;0m" % logging.getLevelName(lvl),
        )

    # inits logger
    logging.basicConfig(
        encoding="utf-8",
        level=config.log_level.value,
        handlers=[logging.StreamHandler()],
    )

    for handler in logging.root.handlers:
        handler.setFormatter(_ZuluFormatter())
