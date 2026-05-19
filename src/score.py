"""Score the champion model and write forecasts to Postgres.

After the champion is published to MLflow, score() generates the next
N-hour forecast and write_forecasts() UPSERTs rows into the `forecasts`
table. The agent + HMI then read those rows via server's REST endpoints
— MLflow stays off the runtime path.
"""

import io
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Final

import pandas as pd
import polars as pl
import psycopg2
from lightgbm import LGBMRegressor
from prophet import Prophet
from xgboost.sklearn import XGBRegressor

from src.config import Config
from src.models import XGBOOST_FEATURE_COLUMNS, PredictiveModels
from src.train import TrainingResult

log = logging.getLogger(__name__)

# Explicit column order — also enforces NOT NULL on every column. Avoids
# `CREATE TABLE LIKE` ambiguity across PG versions re: which constraints
# come along.
_STAGE_DDL: Final[str] = """
    CREATE TEMP TABLE _fcast_stage (
        forecast_for  TIMESTAMPTZ,
        site_id       TEXT,
        measurement   TEXT,
        unit          TEXT,
        value         DOUBLE PRECISION,
        model_name    TEXT,
        model_version INT,
        forecasted_at TIMESTAMPTZ
    ) ON COMMIT DROP
"""

_FORECASTS_DDL: Final[str] = """
    CREATE TABLE IF NOT EXISTS forecasts (
        forecast_for  TIMESTAMPTZ NOT NULL,
        site_id       TEXT NOT NULL,
        measurement   TEXT NOT NULL,
        unit          TEXT NOT NULL,
        value         DOUBLE PRECISION NOT NULL,
        model_name    TEXT NOT NULL,
        model_version INT NOT NULL,
        forecasted_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (forecast_for, site_id, measurement, model_name)
    )
"""


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(os.environ["TIMESERIES_URL"])


def _ensure_forecasts_table() -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(_FORECASTS_DDL)
        conn.commit()


def _future_hours(horizon_hours: int) -> list[datetime]:
    """Hourly UTC timestamps from top-of-next-hour through horizon_hours."""
    start = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) + timedelta(
        hours=1
    )
    return [start + timedelta(hours=i) for i in range(horizon_hours)]


def _score_prophet(model: Prophet, future_ts: list[datetime]) -> list[float]:
    """Prophet predicts a tz-naive DF; strip tz before predict."""
    df = pd.DataFrame({"ds": [ts.replace(tzinfo=None) for ts in future_ts]})
    forecast = model.predict(df)
    return forecast["yhat"].astype(float).tolist()


def _load_history_for_lags() -> pl.DataFrame:
    """Pull the last 7 days of (ts, value) from timeseries_data.

    Needed by `_score_tree` to populate lag_24h + lag_168h features
    for the forecast horizon. 7 days is the longest lag (168h).
    """
    with _connect() as conn:
        return pl.read_database(
            "SELECT ts, value FROM timeseries_data " "ORDER BY ts DESC LIMIT 200",
            connection=conn,
        ).sort("ts")


def _score_tree(
    model: XGBRegressor | LGBMRegressor,
    future_ts: list[datetime],
    history: pl.DataFrame,
) -> list[float]:
    """Predict tree model on time + lagged-value features.

    Builds feature rows by joining each future_ts to its lag values
    from `history` (the most recent actuals). Falls back to the most
    recent available actual if exact-ts lookup misses (won't happen
    with hourly data + a fresh ETL run, but defensive).
    """
    rows: list[dict[str, float | int | datetime]] = []
    hist_by_ts = {row["ts"]: row["value"] for row in history.iter_rows(named=True)}
    most_recent_value = history["value"][-1] if not history.is_empty() else 0.0
    for ts in future_ts:
        lag_24 = hist_by_ts.get(ts - timedelta(hours=24), most_recent_value)
        lag_168 = hist_by_ts.get(ts - timedelta(hours=168), most_recent_value)
        rows.append(
            {
                "ts": ts,
                "hour": ts.hour,
                "day": ts.day,
                "month": ts.month,
                "year": ts.year,
                "dayofweek": ts.weekday(),
                "dayofyear": ts.timetuple().tm_yday,
                "weekofyear": int(ts.isocalendar()[1]),
                "value_lag_24h": float(lag_24),
                "value_lag_168h": float(lag_168),
            }
        )
    feat_df = pl.DataFrame(rows)
    x = feat_df.select(XGBOOST_FEATURE_COLUMNS).to_numpy()
    return [float(v) for v in model.predict(x)]


def score(config: Config, result: TrainingResult) -> pl.DataFrame:
    """Generate horizon forecasts from the champion model.

    Returns:
        DataFrame with (forecast_for: datetime[us, UTC], value: float).
    """
    future_ts = _future_hours(config.forecast_horizon_hours)
    match result.champion:
        case PredictiveModels.PROPHET:
            values = _score_prophet(result.prophet_model, future_ts)
        case PredictiveModels.XGBOOST:
            values = _score_tree(
                result.xgboost_model, future_ts, _load_history_for_lags()
            )
        case PredictiveModels.LIGHTGBM:
            values = _score_tree(
                result.lightgbm_model, future_ts, _load_history_for_lags()
            )
    return pl.DataFrame({"forecast_for": future_ts, "value": values})


def write_forecasts(
    df: pl.DataFrame,
    config: Config,
    model_version: int,
) -> None:
    """UPSERT forecast rows via COPY + temp + INSERT...ON CONFLICT.

    Same pattern as process.load() — COPY is orders of magnitude faster
    than row-by-row INSERT over a remote Postgres, even for the tiny
    horizon row counts.
    """
    _ensure_forecasts_table()
    if df.is_empty():
        return
    forecasted_at = datetime.now(UTC).isoformat()
    buf = io.StringIO()
    for row in df.iter_rows(named=True):
        buf.write(
            "\t".join(
                [
                    row["forecast_for"].isoformat(),
                    config.settlement_point,
                    config.forecast_measurement,
                    config.forecast_unit,
                    str(row["value"]),
                    config.mlflow_model_name,
                    str(model_version),
                    forecasted_at,
                ]
            )
            + "\n"
        )
    buf.seek(0)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(_STAGE_DDL)
        cur.copy_from(
            buf,
            "_fcast_stage",
            sep="\t",
            columns=(
                "forecast_for",
                "site_id",
                "measurement",
                "unit",
                "value",
                "model_name",
                "model_version",
                "forecasted_at",
            ),
        )
        cur.execute(
            "INSERT INTO forecasts "
            "(forecast_for, site_id, measurement, unit, value, model_name, "
            " model_version, forecasted_at) "
            "SELECT forecast_for, site_id, measurement, unit, value, model_name, "
            "       model_version, forecasted_at FROM _fcast_stage "
            "ON CONFLICT (forecast_for, site_id, measurement, model_name) "
            "DO UPDATE SET "
            "  value = EXCLUDED.value, "
            "  unit = EXCLUDED.unit, "
            "  model_version = EXCLUDED.model_version, "
            "  forecasted_at = EXCLUDED.forecasted_at"
        )
        conn.commit()
    log.info(
        "📈 wrote %d forecast rows site=%s measurement=%s",
        df.height,
        config.settlement_point,
        config.forecast_measurement,
    )
