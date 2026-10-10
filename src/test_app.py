"""Unit tests for startup orchestration (MLflow health probe)."""

import logging

import pook
import pytest

from src.app import check_mlflow_health
from src.config import Config, LogLevel


def _fake_config(mlflow_tracking_uri: str) -> Config:
    """Minimal Config for exercising check_mlflow_health() — no DB touched."""
    return Config(
        log_level=LogLevel.INFO,
        site_id="test",
        iso="ercot",
        settlement_point="HB_NORTH",
        gridstatus_dataset="ercot_spp_day_ahead_hourly",
        backfill_start_date="2026-01-01",
        metrics_port=9091,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_model_name="test-model",
        schedule_time="14:00",
        forecast_measurement="dam_lmp_price",
        forecast_unit="usd_per_mwh",
        forecast_horizon_hours=24,
        load_forecast_dataset="ercot_load_forecast_by_forecast_zone",
        load_forecast_zone="north",
    )


def test_check_mlflow_health_logs_uri_on_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reachable MLflow -> logs the exact readiness line platform greps for."""
    # Arrange
    uri = "http://mlflow:5000"
    pook.get(f"{uri}/health").reply(200).body("OK")
    pook.on()

    # Act
    try:
        with caplog.at_level(logging.INFO):
            check_mlflow_health(_fake_config(uri))
    finally:
        pook.off()
        pook.reset()

    # Assert
    assert f"✅ MLflow healthy at {uri}" in caplog.text


def test_check_mlflow_health_raises_on_non_200() -> None:
    """Bad MLflow address -> raise at boot, don't wait for the scheduled run."""
    # Arrange
    uri = "http://mlflow-down:5000"
    pook.get(f"{uri}/health").reply(500).body("error")
    pook.on()

    # Act / Assert
    try:
        with pytest.raises(Exception, match="500"):
            check_mlflow_health(_fake_config(uri))
    finally:
        pook.off()
        pook.reset()
