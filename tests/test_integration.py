"""Integration tests with testcontainers."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import mlflow
import pandas as pd
import pook
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import search_runs
import psycopg2
import requests
from prometheus_client.parser import text_string_to_metric_families

from src.app import app
from src.config import load_config
from src.models import PredictiveModels
from tests.fixtures.containers import start_mlflow, start_postgres


def create_timeseries_table(timeseries_url: str) -> psycopg2.extensions.connection:
    """Create timeseries_data table and return connection.

    Args:
        timeseries_url: PostgreSQL connection string

    Returns:
        Database connection (caller must close)
    """
    conn = psycopg2.connect(timeseries_url)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE timeseries_data (
            ts TIMESTAMPTZ PRIMARY KEY,
            value DOUBLE PRECISION NOT NULL
        )
        """)
    cursor.close()
    return conn


def seed_trending_data(conn: psycopg2.extensions.connection, days: int = 60) -> None:
    """Seed deterministic trending hourly data.

    Real ETL produces hourly DAM SPP rows. Tree-model lag features
    (`value_lag_24h`, `value_lag_168h`) need at least 168 hourly rows
    to populate, so 60 daily rows wouldn't work. 60 days * 24h = 1440
    rows is comfortably above the lag horizon.
    """
    cursor = conn.cursor()
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    total_hours = days * 24
    for i in range(total_hours):
        ts = now - timedelta(hours=total_hours - i)
        value = 50000 + (i * 5)
        cursor.execute(
            "INSERT INTO timeseries_data (ts, value) VALUES (%s, %s)",
            (ts, value),
        )
    conn.commit()
    cursor.close()


@contextmanager
def mock_api(dataset: str) -> Generator[None]:
    """Mock gridstatus.io REST + enable network for testcontainers.

    Shape: JSON `array-of-arrays` — data[0]=column headers, data[1:]=rows.
    `meta.hasNextPage=False` so the pager stops after one fetch.

    Args:
        dataset: gridstatus dataset name (e.g. "ercot_spp_day_ahead_hourly")
    """
    os.environ.setdefault("GRIDSTATUS_API_KEY", "test-key-not-real")
    ts_now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    payload = {
        "data": [
            [
                "interval_start_utc",
                "interval_end_utc",
                "location",
                "location_type",
                "market",
                "spp",
            ],
            [ts_now, ts_now, "HB_NORTH", "Trading Hub", "DAY_AHEAD_HOURLY", 42.5],
        ],
        "meta": {"hasNextPage": False, "cursor": None},
    }
    pook.get(f"https://api.gridstatus.io/v1/datasets/{dataset}/query").persist().reply(
        200
    ).json(payload)
    pook.enable_network("localhost", "127.0.0.1")
    pook.on()
    try:
        yield
    finally:
        pook.off()
        pook.reset()


def parse_prometheus_metrics(metrics_text: str) -> dict[str, float]:
    """Parse Prometheus metrics text into dict.

    Args:
        metrics_text: Prometheus text format metrics

    Returns:
        Dict mapping metric name to value
    """
    metric_families = text_string_to_metric_families(metrics_text)
    metrics = {}
    for family in metric_families:
        for sample in family.samples:
            metrics[sample.name] = sample.value
    return metrics


class TestIntegration:
    """Integration tests for full ML pipeline."""

    def test_publish_model_and_verify_inference(self) -> None:
        """Test happy path: app trains model, publishes to MLflow, inference works.

        Integration test: Insert test data directly, run app(), verify MLflow inference.
        """
        # Arrange
        with (
            start_mlflow() as mlflow_c,
            start_postgres(image="timescale/timescaledb:latest-pg15") as pg,
        ):
            test_config = load_config().model_copy(
                update={
                    "mlflow_tracking_uri": mlflow_c.url,
                    "mlflow_model_name": "test_timeseries_predictor",
                    "prophet_daily_seasonality": False,
                    "prophet_yearly_seasonality": False,
                }
            )

            # Seed database with trending data
            os.environ["TIMESERIES_URL"] = pg.url
            conn = create_timeseries_table(pg.url)
            try:
                seed_trending_data(conn)
            finally:
                conn.close()

            # Act - run full pipeline with mocked API
            with mock_api(test_config.gridstatus_dataset):
                app(test_config, mode="once")

            # Assert - verify model registered and inference works
            mlflow.set_tracking_uri(mlflow_c.url)
            model_versions = MlflowClient().search_model_versions(
                "name='test_timeseries_predictor'"
            )
            assert len(model_versions) > 0

            # Load model and run inference
            model_uri = f"models:/test_timeseries_predictor/{model_versions[0].version}"
            loaded_model = mlflow.pyfunc.load_model(model_uri)

            # Get champion model type from MLflow
            runs = search_runs()
            assert isinstance(runs, pd.DataFrame)
            champion_str = runs["params.champion_model"].iloc[0]
            champion = PredictiveModels(champion_str)

            # Create appropriate input based on champion model type
            future_dates = pd.date_range(start=pd.Timestamp.now(), periods=7, freq="D")

            match champion:
                case PredictiveModels.PROPHET:
                    predictions = loaded_model.predict(
                        pd.DataFrame({"ds": future_dates})
                    )
                case PredictiveModels.XGBOOST:
                    test_df = pd.DataFrame(
                        {
                            "hour": future_dates.hour,
                            "day": future_dates.day,
                            "month": future_dates.month,
                            "year": future_dates.year,
                            "dayofweek": future_dates.dayofweek,
                            "dayofyear": future_dates.dayofyear,
                            "weekofyear": future_dates.isocalendar().week.astype(int),
                        }
                    )
                    predictions = loaded_model.predict(test_df)
                case _:
                    raise ValueError(f"Unknown champion model type: {champion}")

            # Verify predictions
            assert predictions is not None
            assert len(predictions) == 7

            # Assert forecasts table populated by score+write step
            verify_conn = psycopg2.connect(pg.url)
            try:
                verify_cur = verify_conn.cursor()
                verify_cur.execute(
                    "SELECT COUNT(*), MIN(site_id), MIN(measurement), MIN(unit) "
                    "FROM forecasts"
                )
                row = verify_cur.fetchone()
                assert row is not None
                count, site_id, measurement, unit = row
                assert count == test_config.forecast_horizon_hours
                assert site_id == test_config.settlement_point
                assert measurement == test_config.forecast_measurement
                assert unit == test_config.forecast_unit
                verify_cur.close()
            finally:
                verify_conn.close()

    def test_metrics_endpoint_scrapable_after_run(self) -> None:
        """Embedded /metrics serves gauges after run_pipeline updates them.

        Simulates the real prod path: `start_metrics_server` binds a port,
        `run_pipeline` runs end-to-end (ETL + train + publish + score),
        then a Prom-style scrape against `localhost:<port>/metrics`
        returns the gauge values. Same shape Prom would see in `--mode
        schedule`.
        """
        from src.app import run_pipeline
        from src.metrics import start_metrics_server

        # Pick a high free port that won't collide with the local prom default.
        metrics_port = 19090
        with (
            start_mlflow() as mlflow_c,
            start_postgres(image="timescale/timescaledb:latest-pg15") as pg,
        ):
            test_config = load_config().model_copy(
                update={
                    "mlflow_tracking_uri": mlflow_c.url,
                    "mlflow_model_name": "test_metrics_scrape_predictor",
                    "prophet_daily_seasonality": False,
                    "prophet_yearly_seasonality": False,
                    "metrics_port": metrics_port,
                }
            )

            # Seed deterministic trending data so training reliably succeeds.
            os.environ["TIMESERIES_URL"] = pg.url
            conn = create_timeseries_table(pg.url)
            try:
                seed_trending_data(conn)
            finally:
                conn.close()

            # Act — start the metrics server then run pipeline.
            start_metrics_server(metrics_port)
            with mock_api(test_config.gridstatus_dataset):
                run_pipeline(test_config)

            # Assert — scrape the gauge endpoint, expect all 5 gauges present.
            response = requests.get(
                f"http://localhost:{metrics_port}/metrics", timeout=10
            )
            assert response.status_code == 200
            metrics = parse_prometheus_metrics(response.text)
            for name in (
                "champion_mae",
                "baseline_mae",
                "challenger_mae",
                "champion_model",
                "model_version",
            ):
                assert name in metrics, f"missing gauge {name}"
            # Champion encoded as 1 (Prophet) or 2 (XGBoost) — both valid.
            assert metrics["champion_model"] in (1.0, 2.0)
            # Model version is the just-published one, starts at 1.
            assert metrics["model_version"] >= 1.0
