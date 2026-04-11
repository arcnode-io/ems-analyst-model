"""Application orchestration with scheduling and MLflow publishing."""

import logging
import time

import mlflow
import pandas as pd
import schedule

from src.config import Config
from src.models import SCHEDULE_POLL_INTERVAL_SECONDS, PredictiveModels
from src.process import process
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

    with mlflow.start_run():
        # Log metrics
        champion_mae = get_model_mae(
            result.champion, result.prophet_mae, result.xgboost_mae
        )
        mlflow.log_metric("champion_mae", champion_mae)
        mlflow.log_metric("prophet_mae", result.prophet_mae)
        mlflow.log_metric("xgboost_mae", result.xgboost_mae)
        mlflow.log_metric("baseline_mae", result.baseline_mae)

        # Log parameters
        mlflow.log_param("champion_model", result.champion.value)

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
        client = mlflow.MlflowClient()
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
        logging.info("Starting ML pipeline")

        # ETL
        process(config)

        # Train models
        result = train_models(config)

        # Publish to MLflow
        version = publish_to_mlflow(config, result)
        logging.info(f"Published {result.champion.value} model as {version}")

        logging.info("ML pipeline completed successfully")
    except Exception:
        logging.exception("Pipeline failed")
        raise


def app(config: Config, mode: str = "once") -> None:
    """Run the application.

    Args:
        config: Configuration object
        mode: "once" to run once, "schedule" for daily runs
    """
    if mode == "schedule":
        schedule.every().day.at(config.schedule_time).do(run_pipeline, config=config)
        logging.info(f"Scheduler started. Will run daily at {config.schedule_time}")

        while True:
            schedule.run_pending()
            time.sleep(SCHEDULE_POLL_INTERVAL_SECONDS)
    else:
        run_pipeline(config)
