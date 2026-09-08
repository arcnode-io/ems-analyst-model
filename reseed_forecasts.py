"""Re-time the canonical champion forecast curve onto the live horizon.

One-shot demo maintenance. gridstatus credits are exhausted until the
monthly reset, so there's no fresh ERCOT SPP to score. Instead this reads
the shipped champion curve (`demo_data/champion_dam_lmp_curve.csv` — the
2026-05-19 v4 LightGBM 24h forecast) and re-lays it on `now -> now +
horizon`, aligned by UTC hour-of-day. Same re-timing trick the demo
historian path uses for `/measurements`.

Rows are written under the configured model_name at CHAMPION_VERSION, so
the chart's lineage label stays truthful. A later real pipeline run
UPSERTs over these by (forecast_for, ...). Safe to re-run — do so shortly
before recording so the 24-point curve is complete.

Run:
    uv run --env-file secrets.env python reseed_forecasts.py
"""

import logging
from pathlib import Path
from typing import Final

import polars as pl

from src.config import load_config
from src.score import _future_hours, realign_curve_by_hour, write_forecasts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_CURVE_CSV: Final[Path] = (
    Path(__file__).parent / "demo_data" / "champion_dam_lmp_curve.csv"
)
# The champion curve was produced by dam-lmp-forecast-ercot-hb-north v4.
CHAMPION_VERSION: Final[int] = 4


def main() -> None:
    """Load the champion curve, re-time it, write it to the live horizon."""
    cfg = load_config()
    curve = pl.read_csv(_CURVE_CSV)
    target_hours = _future_hours(cfg.forecast_horizon_hours)
    realigned = realign_curve_by_hour(curve, target_hours)
    write_forecasts(realigned, cfg, CHAMPION_VERSION)
    log.info(
        "reseeded %d rows %s v%d covering %s -> %s",
        realigned.height,
        cfg.mlflow_model_name,
        CHAMPION_VERSION,
        target_hours[0].isoformat(),
        target_hours[-1].isoformat(),
    )


if __name__ == "__main__":
    main()
