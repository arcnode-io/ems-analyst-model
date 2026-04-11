"""Integration tests with testcontainers."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import mlflow
import pandas as pd
import pook
import psycopg2
import requests
from prometheus_client.parser import text_string_to_metric_families

from src.app import app
from src.config import Config, load_config
from src.models import PredictiveModels
from tests.fixtures.containers import start_mlflow, start_postgres, start_pushgateway


def create_timeseries_table(
    config: Config, password: str
) -> psycopg2.extensions.connection:
    """Create timeseries_data table and return connection.

    Args:
        config: Configuration with DB connection details
        password: Database password

    Returns:
        Database connection (caller must close)
    """
    conn = psycopg2.connect(
        host=config.db_host,
        database=config.db_name,
        user=config.db_user,
        password=password,
        port=config.db_port,
    )
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE timeseries_data (
            timestamp TIMESTAMPTZ PRIMARY KEY,
            value DOUBLE PRECISION NOT NULL
        )
        """)
    cursor.close()
    return conn


def seed_trending_data(conn: psycopg2.extensions.connection, days: int = 60) -> None:
    """Seed database with deterministic trending test data.

    Args:
        conn: Database connection
        days: Number of days of data to generate
    """
    cursor = conn.cursor()
    now = datetime.now(UTC)
    for i in range(days):
        timestamp = now - timedelta(days=days - i)
        value = 50000 + (i * 200)
        cursor.execute(
            "INSERT INTO timeseries_data (timestamp, value) VALUES (%s, %s)",
            (timestamp, value),
        )
    conn.commit()
    cursor.close()


def seed_random_data(conn: psycopg2.extensions.connection, days: int = 60) -> None:
    """Seed database with random data (causes model degradation).

    Random data = high MAE for models, low MAE for baseline (mean).

    Args:
        conn: Database connection
        days: Number of days of data to generate
    """
    import random

    cursor = conn.cursor()
    now = datetime.now(UTC)
    random.seed(42)
    values = [random.uniform(45000, 55000) for _ in range(days)]  # noqa: S311
    for i in range(days):
        timestamp = now - timedelta(days=days - i)
        cursor.execute(
            "INSERT INTO timeseries_data (timestamp, value) VALUES (%s, %s)",
            (timestamp, values[i]),
        )
    conn.commit()
    cursor.close()


@contextmanager
def mock_api(api_url: str) -> Generator[None]:
    """Mock API and enable network for testcontainers.

    Args:
        api_url: API URL to mock

    Yields:
        None
    """
    mock_response = {"prices": [[int(datetime.now(UTC).timestamp() * 1000), 50000.0]]}
    pook.get(api_url).reply(200).json(mock_response)
    pook.enable_network("localhost", "127.0.0.1")
    pook.on()
    try:
        yield
    finally:
        pook.off()


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
        password = os.environ["POSTGRES_PASSWORD"]

        with (
            start_mlflow() as mlflow_c,
            start_postgres(
                password=password, image="timescale/timescaledb:latest-pg15"
            ) as pg,
        ):
            test_config = load_config().model_copy(
                update={
                    "db_port": pg.port,
                    "mlflow_tracking_uri": mlflow_c.url,
                    "mlflow_model_name": "test_timeseries_predictor",
                    "prophet_daily_seasonality": False,
                    "prophet_yearly_seasonality": False,
                }
            )

            # Seed database with trending data
            conn = create_timeseries_table(test_config, password)
            try:
                seed_trending_data(conn)
            finally:
                conn.close()

            # Act - run full pipeline with mocked API
            with mock_api(test_config.api_url):
                app(test_config, mode="once")

            # Assert - verify model registered and inference works
            mlflow.set_tracking_uri(mlflow_c.url)
            model_versions = mlflow.MlflowClient().search_model_versions(
                "name='test_timeseries_predictor'"
            )
            assert len(model_versions) > 0

            # Load model and run inference
            model_uri = f"models:/test_timeseries_predictor/{model_versions[0].version}"
            loaded_model = mlflow.pyfunc.load_model(model_uri)

            # Get champion model type from MLflow
            runs = mlflow.search_runs()
            champion_str = runs["params.champion_model"].iloc[0]  # type: ignore[call-overload]
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

    def test_pushes_metrics_via_prometheus(self) -> None:
        """Test that MAE metrics are pushed to Prometheus Pushgateway on degradation.

        Integration test: Run app with data causing degradation (baseline > champion),
        verify push_model_metrics sends metrics to Pushgateway.
        """
        # Arrange
        password = os.environ["POSTGRES_PASSWORD"]

        with (
            start_mlflow() as mlflow_c,
            start_postgres(
                password=password, image="timescale/timescaledb:latest-pg15"
            ) as pg,
            start_pushgateway() as pushgateway,
        ):
            test_config = load_config().model_copy(
                update={
                    "db_port": pg.port,
                    "mlflow_tracking_uri": mlflow_c.url,
                    "mlflow_model_name": "test_degradation_predictor",
                    "prophet_daily_seasonality": False,
                    "prophet_yearly_seasonality": False,
                    "prometheus_pushgateway": f"localhost:{pushgateway.port}",
                }
            )

            # Seed database with random data (triggers degradation)
            conn = create_timeseries_table(test_config, password)
            try:
                seed_random_data(conn)
            finally:
                conn.close()

            # Act - run full pipeline with mocked API (triggers degradation detection)
            with mock_api(test_config.api_url):
                # First run trains initial model (no current_champion)
                app(test_config, mode="once")

                # Second run with current_champion set - triggers degradation detection
                from src.train import train_models

                train_models(test_config, current_champion=PredictiveModels.PROPHET)

            # Assert - verify metrics were pushed to Pushgateway
            response = requests.get(
                f"http://localhost:{pushgateway.port}/metrics", timeout=10
            )
            assert response.status_code == 200

            metrics = parse_prometheus_metrics(response.text)

            # Verify degradation metrics are present
            assert "champion_mae" in metrics
            assert "challenger_mae" in metrics
            assert "champion_model" in metrics
            assert metrics["champion_model"] == 1.0  # PROPHET
