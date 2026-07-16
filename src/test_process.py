"""Unit tests for the gridstatus → timeseries_data transform."""

import httpx
import polars as pl
import pytest

from src import process as process_mod
from src.config import Config, LogLevel
from src.process import process, transform


def test_transform_picks_ts_and_spp_from_gridstatus_shape() -> None:
    """Transform projects gridstatus CSV rows to (ts, value) and dedupes."""
    # Arrange — pl.read_csv yields String columns; transform parses to datetime
    raw = pl.DataFrame(
        {
            "interval_start_utc": [
                "2026-05-17T00:00:00",
                "2026-05-17T01:00:00",
                "2026-05-17T00:00:00",  # duplicate — transform should dedupe
            ],
            "location": ["HB_NORTH", "HB_NORTH", "HB_NORTH"],
            "spp": [42.5, -12.3, 42.5],  # negative LMP is valid (oversupply)
        }
    )

    # Act
    actual = transform(raw)

    # Assert
    assert isinstance(actual, pl.DataFrame)
    assert len(actual) == 2  # dedup'd
    assert actual.columns == ["ts", "value"]
    assert actual["value"][0] == 42.5
    assert actual["value"][1] == -12.3  # negative price preserved


def test_transform_empty_input_returns_empty_df() -> None:
    """Empty extract (no new rows) round-trips to empty TimeseriesData."""
    # Arrange
    empty = pl.DataFrame(
        schema={
            "interval_start_utc": pl.Utf8,
            "location": pl.Utf8,
            "spp": pl.Float64,
        }
    )

    # Act
    actual = transform(empty)

    # Assert
    assert actual.is_empty()
    assert actual.columns == ["ts", "value"]


def _fake_config() -> Config:
    """Minimal Config for exercising `process()` — no DB touched."""
    return Config(
        log_level=LogLevel.INFO,
        iso="ercot",
        settlement_point="HB_NORTH",
        gridstatus_dataset="ercot_spp_day_ahead_hourly",
        backfill_start_date="2026-01-01",
        metrics_port=9091,
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_model_name="test-model",
        schedule_time="02:00",
        forecast_measurement="dam_lmp_price",
        forecast_unit="usd_per_mwh",
        forecast_horizon_hours=24,
        load_forecast_dataset="ercot_load_forecast_by_forecast_zone",
        load_forecast_zone="north",
    )


def test_process_swallows_403_from_spp_extract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """403 from gridstatus during SPP extract must not crash the scheduler.

    Mirrors the existing load-forecast graceful-skip pattern. The
    load-forecast half still runs; here we replace it with a no-op so
    the test doesn't need a DB.
    """

    # Arrange
    def _raise_403(_config: Config) -> pl.DataFrame:
        request = httpx.Request("GET", "https://api.gridstatus.io/v1/x")
        response = httpx.Response(403, request=request)
        raise httpx.HTTPStatusError("quota", request=request, response=response)

    monkeypatch.setattr(process_mod, "extract", _raise_403)
    monkeypatch.setattr(process_mod, "_process_load_forecast", lambda _c: None)

    # Act — must not raise
    process(_fake_config())

    # Assert — reaching here IS the assertion (no exception propagated).
    assert True
