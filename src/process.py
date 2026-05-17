"""ETL — pull ERCOT DAM SPP from gridstatus.io REST into timeseries_data.

Targets one location (settlement point) at a time so the training pipeline
gets a single (ts, value) time series. First run: backfills from
`backfill_start_date`. Subsequent runs: pulls latest_ts + 1h -> now,
idempotent via ON CONFLICT.

REST direct (no SDK) to avoid the gridstatusio numpy<2 constraint that
conflicts with our numpy>=2 stack. Auth via `x-api-key` header so the
key never lands in URL query-string logs.

GRIDSTATUS_API_KEY env var feeds the header.
"""

import io
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import httpx
import pandera.polars as pa
import polars as pl
import psycopg2

from src.config import Config
from src.models import TimeseriesData

log = logging.getLogger(__name__)

_GRIDSTATUS_BASE: Final[str] = "https://api.gridstatus.io/v1"
_PAGE_SIZE: Final[int] = 10_000
_HTTP_TIMEOUT: Final[float] = 60.0


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(os.environ["TIMESERIES_URL"])


def _ensure_table() -> None:
    """Idempotent CREATE TABLE — same shape the trainer expects."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS timeseries_data (
                ts    TIMESTAMPTZ PRIMARY KEY,
                value DOUBLE PRECISION NOT NULL
            )
        """)
        conn.commit()


def _latest_ts() -> datetime | None:
    """Most-recent ts already in the table (or None if empty)."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT MAX(ts) FROM timeseries_data")
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def extract(config: Config) -> pl.DataFrame:
    """Pull hourly DAM SPP for the configured location.

    First-run: backfills from config.backfill_start_date to now.
    Subsequent runs: pulls from latest_ts + 1h to now (incremental).
    """
    _ensure_table()
    latest = _latest_ts()
    end = datetime.now(UTC)
    if latest is None:
        start = datetime.strptime(config.backfill_start_date, "%Y-%m-%d").replace(
            tzinfo=UTC
        )
        log.info("first run — backfilling from %s", start.isoformat())
    else:
        start = latest + timedelta(hours=1)
        log.info("incremental pull from %s", start.isoformat())
    if start >= end:
        log.info("no new data to pull (start >= end)")
        return pl.DataFrame(
            schema={"ts": pl.Datetime("us", "UTC"), "value": pl.Float64}
        )
    return _fetch_dataset(
        dataset=config.gridstatus_dataset,
        filter_column="location",
        filter_value=config.settlement_point,
        start=start,
        end=end,
    )


def _fetch_dataset(
    dataset: str,
    filter_column: str,
    filter_value: str,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """Page through gridstatus REST returning a stitched polars DataFrame."""
    headers = {"x-api-key": os.environ["GRIDSTATUS_API_KEY"]}
    url = f"{_GRIDSTATUS_BASE}/datasets/{dataset}/query"
    frames: list[pl.DataFrame] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {
            "start_time": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "filter_column": filter_column,
            "filter_value": filter_value,
            "limit": _PAGE_SIZE,
            "return_format": "csv",
        }
        if cursor:
            params["cursor"] = cursor
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            body = resp.text
            cursor = resp.headers.get("x-next-page-cursor")
        if body.strip():
            frames.append(pl.read_csv(io.StringIO(body)))
        if not cursor:
            break
    if not frames:
        return pl.DataFrame(
            schema={"ts": pl.Datetime("us", "UTC"), "value": pl.Float64}
        )
    return pl.concat(frames)


@pa.check_types
def transform(raw: pl.DataFrame) -> TimeseriesData:
    """Pick (interval_start_utc, spp) -> (ts, value). Drop dupes."""
    if raw.is_empty():
        return TimeseriesData(
            pl.DataFrame(schema={"ts": pl.Datetime("us", "UTC"), "value": pl.Float64})
        )
    return TimeseriesData(
        raw.select(
            pl.col("interval_start_utc").str.to_datetime(time_zone="UTC").alias("ts"),
            pl.col("spp").cast(pl.Float64).alias("value"),
        )
        .unique(subset=["ts"])
        .sort("ts")
    )


@pa.check_types
def load(df: TimeseriesData, _config: Config) -> None:
    """Upsert into timeseries_data — ON CONFLICT keeps latest value."""
    if df.is_empty():
        return
    with _connect() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO timeseries_data (ts, value) VALUES (%s, %s) "
            "ON CONFLICT (ts) DO UPDATE SET value = EXCLUDED.value",
            [(row["ts"], row["value"]) for row in df.iter_rows(named=True)],
        )
        conn.commit()
    log.info("loaded %d rows into timeseries_data", df.height)


def process(config: Config) -> None:
    """Execute ETL: extract → transform → load."""
    raw = extract(config)
    clean = transform(raw)
    load(clean, config)
