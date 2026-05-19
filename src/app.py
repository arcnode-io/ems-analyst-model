"""Application orchestration with scheduling and MLflow publishing."""

import logging
import time

import mlflow
import pandas as pd
import schedule
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import log_metric, log_param, start_run

from src.config import Config
from src.metrics import start_metrics_server, update_gauges
from src.models import SCHEDULE_POLL_INTERVAL_SECONDS, PredictiveModels
from src.process import process
from src.score import score, write_forecasts
from src.train import TrainingResult, get_model_mae, train_models
from src.utils import create_xgboost_input_example


def publish_to_mlflow(config: Config, result: TrainingResult) -> str:
    """Publish champion model to MLflow with auto-versioning.

    Args:
        config: Configuration object
        result: Training result with champion model

    Returns:
        Model version string
    """
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    with start_run():
        # Log metrics
        champion_mae = get_model_mae(
            result.champion, result.prophet_mae, result.xgboost_mae
        )
        log_metric("champion_mae", champion_mae)
        log_metric("prophet_mae", result.prophet_mae)
        log_metric("xgboost_mae", result.xgboost_mae)
        log_metric("baseline_mae", result.baseline_mae)

        # Log parameters
        log_param("champion_model", result.champion.value)

        # Log champion model with input_example for signature inference
        if result.champion == PredictiveModels.PROPHET:
            # Prophet expects DataFrame with 'ds' column (datetime)
            input_example = pd.DataFrame({"ds": [pd.Timestamp.now()]})
            mlflow.prophet.log_model(
                result.prophet_model,
                name="model",
                registered_model_name=config.mlflow_model_name,
                input_example=input_example,
            )
        else:
            # XGBoost expects DataFrame with time-based features
            input_example = create_xgboost_input_example(pd.Timestamp.now())
            mlflow.xgboost.log_model(
                result.xgboost_model,
                name="model",
                registered_model_name=config.mlflow_model_name,
                input_example=input_example,
            )

        # Get version
        client = MlflowClient()
        model_versions = client.search_model_versions(
            f"name='{config.mlflow_model_name}'"
        )
        latest_version = max(int(v.version) for v in model_versions)

        return f"v{latest_version}"


def run_pipeline(config: Config) -> None:
    """Run full pipeline: ETL → train → publish.

    Args:
        config: Configuration object
    """
    try:
        logging.info("🚀 ML pipeline starting")

        # ETL
        process(config)

        # Train models
        logging.info("🤖 training Prophet + XGBoost")
        result = train_models(config)
        logging.info(
            "🏆 champion: %s (prophet_mae=%.3f xgboost_mae=%.3f baseline_mae=%.3f)",
            result.champion.value,
            result.prophet_mae,
            result.xgboost_mae,
            result.baseline_mae,
        )

        # Publish to MLflow
        logging.info("📈 publishing to MLflow at %s", config.mlflow_tracking_uri)
        version = publish_to_mlflow(config, result)
        logging.info(
            "✨ published %s model as %s → models:/%s/%s",
            result.champion.value,
            version,
            config.mlflow_model_name,
            version,
        )

        # Score the champion and write forecasts to Postgres for the
        # agent + HMI to read via server's REST endpoints.
        logging.info("🔮 scoring champion for next %dh", config.forecast_horizon_hours)
        forecasts_df = score(config, result)
        version_int = int(version.lstrip("v"))
        write_forecasts(forecasts_df, config, version_int)

        # Embedded /metrics gauges — Prom scrapes these directly in --mode
        # schedule. No pushgateway needed; the daemon process is always-up.
        update_gauges(result, version_int)

        logging.info("🎉 ML pipeline completed successfully")
    except Exception:
        logging.exception("💥 Pipeline failed")
        raise


def app(config: Config, mode: str = "once") -> None:
    """Run the application.

    Args:
        config: Configuration object
        mode: "once" to run once, "schedule" for daily runs
    """
    if mode == "schedule":
        # Daemon mode → expose Prom /metrics so the always-up process can
        # be scraped directly. --mode once skips this (no need to bind a
        # port for a one-shot CLI run; metrics still land in MLflow).
        start_metrics_server(config.metrics_port)
        schedule.every().day.at(config.schedule_time).do(run_pipeline, config=config)
        logging.info(f"Scheduler started. Will run daily at {config.schedule_time}")

        while True:
            schedule.run_pending()
            time.sleep(SCHEDULE_POLL_INTERVAL_SECONDS)
    else:
        run_pipeline(config)
