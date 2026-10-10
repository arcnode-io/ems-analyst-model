import enum
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel
import yaml

# Leaf value a YAML field can hold — explicit instead of Any/object per
# this repo's typing rules. cfg.yml's stage blocks are flat (no nested
# config sections), so a stage is dict[str, YamlScalar], no recursion
# needed. Reason not recursive: beartype can't runtime-check a
# forward-referencing recursive `type` alias (confirmed via
# BeartypeCallHintForwardRefException) — if a nested config block is
# ever added, this needs a real rethink, not a wider alias.
type YamlScalar = str | int | bool | float | None


class LogLevel(enum.StrEnum):
    """Logging levels for the application."""

    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"
    DEBUG = "DEBUG"


class Config(BaseModel):  # noqa: D101
    log_level: LogLevel
    site_id: str
    iso: str
    settlement_point: str
    gridstatus_dataset: str
    backfill_start_date: str  # YYYY-MM-DD, used for first-run historical pull
    metrics_port: int  # Embedded Prom /metrics port (--mode schedule only)
    mlflow_tracking_uri: str
    mlflow_model_name: str
    schedule_time: str
    prophet_daily_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    prophet_yearly_seasonality: bool = True
    forecast_measurement: str
    forecast_unit: str
    forecast_horizon_hours: int
    # ERCOT load-forecast feature wiring. zone matches one of the
    # column names in ercot_load_forecast_by_forecast_zone
    # (north|south|west|houston|system_total).
    load_forecast_dataset: str
    load_forecast_zone: str


class _ConfigMap(BaseModel):
    local: Config
    beta: Config


def _deep_merge(
    base: dict[str, YamlScalar], overlay: dict[str, YamlScalar]
) -> dict[str, YamlScalar]:
    """Overlay wins per key. Flat merge — cfg.yml's stage blocks have no nested sections.

    "Deep" to match ems_analyst_agent's config loader by name — same
    mechanism, same merge rule — but genuinely flat here since this
    repo's Config has no nested block (unlike the agent's `market`).
    """
    return {**base, **overlay}


def load_config() -> Config:
    """Resolve cfg.yml[ENV] -> Config. ENV picks the stage; CFG_CUSTOMER_PATH merges over it.

    Reason: platform needs to set mlflow_tracking_uri/site_id per deploy
    (cloud/on-prem each point at their own MLflow container, and each
    customer site gets its own site_id) without baking a new image.
    CFG_CUSTOMER_PATH is a *file pointer* env var, not a config value
    itself — the actual values still live in YAML, matching this repo's
    "config in cfg.yml, not env files" rule. Same mechanism as
    ems_analyst_agent's loader, for symmetry across services.

    Returns:
        Config object for the current environment, with any
        CFG_CUSTOMER_PATH overrides applied.

    Raises:
        FileNotFoundError: If cfg.yml file is not found
        yaml.YAMLError: If cfg.yml or the customer override contains invalid YAML

    Example:
        >>> config = load_config()  # Uses ENV=local by default
        >>> config.log_level
        <Level.INFO: 'INFO'>

    """
    with open("cfg.yml") as file:
        raw: dict[str, dict[str, YamlScalar]] = yaml.safe_load(file)
    environment = os.environ.get("ENV", "local")
    customer_path_env = os.environ.get("CFG_CUSTOMER_PATH")
    if customer_path_env:
        customer_path = Path(customer_path_env)
        if customer_path.exists():
            with open(customer_path) as file:
                customer: dict[str, YamlScalar] = yaml.safe_load(file) or {}
            stage_raw = raw.get(environment, raw["local"])
            raw[environment] = _deep_merge(stage_raw, customer)
    config_map = _ConfigMap.model_validate(raw)
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
        level=config.log_level.value,
        handlers=[logging.StreamHandler()],
    )

    for handler in logging.root.handlers:
        handler.setFormatter(_ZuluFormatter())
