"""Utility functions for ML pipeline."""

import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from src.models import XGBOOST_FEATURE_COLUMNS, PredictiveModels


def push_model_metrics(
    pushgateway_url: str,
    champion_mae: float,
    challenger_mae: float,
    champion: PredictiveModels,
) -> None:
    """Push model metrics to Prometheus Pushgateway.

    Args:
        pushgateway_url: URL of the Prometheus Pushgateway
        champion_mae: Champion model MAE
        challenger_mae: Challenger model MAE
        champion: Champion model
    """
    registry = CollectorRegistry()

    # Create gauges
    champion_mae_gauge = Gauge("champion_mae", "Champion model MAE", registry=registry)
    challenger_mae_gauge = Gauge(
        "challenger_mae", "Challenger model MAE", registry=registry
    )
    champion_gauge = Gauge(
        "champion_model", "Current champion model", registry=registry
    )

    # Set metrics
    champion_mae_gauge.set(champion_mae)
    challenger_mae_gauge.set(challenger_mae)
    champion_gauge.set(1 if champion == PredictiveModels.PROPHET else 2)

    # Push to gateway
    push_to_gateway(pushgateway_url, job="ml_training", registry=registry)


def create_xgboost_input_example(timestamp: pd.Timestamp) -> pd.DataFrame:
    """Create XGBoost input example from timestamp.

    Args:
        timestamp: Input timestamp

    Returns:
        DataFrame with time-based features for XGBoost
    """
    return pd.DataFrame(
        {
            col: [
                (
                    getattr(timestamp, col)
                    if col != "weekofyear"
                    else timestamp.isocalendar().week
                )
            ]
            for col in XGBOOST_FEATURE_COLUMNS
        }
    )
