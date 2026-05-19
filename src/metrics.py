"""Embedded Prometheus /metrics endpoint for the model service.

When the model runs in `--mode schedule` it's a long-lived daemon, so
Prom can scrape it directly — no pushgateway needed. Gauges hold the
latest run's values until the next daily training overwrites them.

In `--mode once` we don't start the HTTP server (caller doesn't need
scraping; metric lands in MLflow as permanent record). Gauge updates
still happen so any in-process consumer sees them.
"""

import logging
from typing import Final

from prometheus_client import Gauge, start_http_server

from src.models import PredictiveModels
from src.train import TrainingResult, get_model_mae

log = logging.getLogger(__name__)

_DEFAULT_METRICS_PORT: Final[int] = 9090

# Module-level gauges = singletons across imports. Prom expects a single
# CollectorRegistry per process; start_http_server uses the default one.
_champion_mae = Gauge(
    "champion_mae", "MAE of the deployed champion model on the holdout window"
)
_baseline_mae = Gauge(
    "baseline_mae", "MAE of the naive baseline (mean) on the same window"
)
_challenger_mae = Gauge(
    "challenger_mae", "MAE of the best challenger model on the holdout window"
)
# Encoded as int gauge: 1=Prophet, 2=XGBoost, 3=LightGBM.
_champion_model = Gauge(
    "champion_model",
    "Current champion model encoded (1=Prophet, 2=XGBoost, 3=LightGBM)",
)
_model_version = Gauge(
    "model_version", "MLflow registry version of the currently published model"
)


def start_metrics_server(port: int = _DEFAULT_METRICS_PORT) -> None:
    """Start an HTTP server serving /metrics on the given port.

    Idempotent at the call-site level — should be invoked once per
    process at startup (e.g. before the schedule loop). Re-entry would
    raise OSError(EADDRINUSE).
    """
    start_http_server(port)
    log.info("📊 metrics endpoint serving on :%d/metrics", port)


_CHAMPION_ENCODING: dict[PredictiveModels, int] = {
    PredictiveModels.PROPHET: 1,
    PredictiveModels.XGBOOST: 2,
    PredictiveModels.LIGHTGBM: 3,
}


def _best_challenger_mae(result: TrainingResult) -> float:
    """Lowest MAE among non-champion models."""
    all_maes: dict[PredictiveModels, float] = {
        PredictiveModels.PROPHET: result.prophet_mae,
        PredictiveModels.XGBOOST: result.xgboost_mae,
        PredictiveModels.LIGHTGBM: result.lightgbm_mae,
    }
    return min(v for k, v in all_maes.items() if k != result.champion)


def update_gauges(result: TrainingResult, model_version: int) -> None:
    """Update every gauge from a completed training run."""
    champion_mae = get_model_mae(
        result.champion,
        result.prophet_mae,
        result.xgboost_mae,
        result.lightgbm_mae,
    )
    challenger_mae = _best_challenger_mae(result)
    _champion_mae.set(champion_mae)
    _baseline_mae.set(result.baseline_mae)
    _challenger_mae.set(challenger_mae)
    _champion_model.set(_CHAMPION_ENCODING[result.champion])
    _model_version.set(model_version)
    log.info(
        "📈 gauges: champion=%s mae=%.3f baseline=%.3f challenger=%.3f v=%d",
        result.champion.value,
        champion_mae,
        result.baseline_mae,
        challenger_mae,
        model_version,
    )
