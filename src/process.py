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

import logging
import os
import re
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import httpx
import pandera.polars as pa
import polars as pl
import psycopg2

from src.config import Config
from src.models import TimeseriesData

log = logging.getLogger(__name__)

# Free tier: 1 req/sec, 30/min, 600/hr. Pace at 1.1s/req to stay clear of
# the per-second limit even with network jitter.
_PACE_SECONDS: Final[float] = 1.1
# Retry/backoff numbers mirror gridstatusio SDK defaults (which are
# presumably tuned for what they actually enforce, not just what's
# documented). delay = _BACKOFF_BASE * (_BACKOFF_EXP ** attempt).
_MAX_RETRIES: Final[int] = 5
_BACKOFF_BASE: Final[float] = 2.0
_BACKOFF_EXP: Final[float] = 2.0
_DEFAULT_RETRY_SECONDS: Final[float] = 60.0
# HTTP status codes we retry on (in addition to 429).
_RETRY_STATUS: Final[frozenset[int]] = frozenset({500, 502, 503, 504})


def _parse_retry_seconds(body: str) -> float:
    """Pull 'Try again in N seconds' out of gridstatus 429 JSON detail."""
    m = re.search(r"in\s+(\d+)\s+seconds", body, re.IGNORECASE)
    return float(m.group(1)) if m else _DEFAULT_RETRY_SECONDS


def _backoff_wait(attempt: int, suggested: float = 0.0) -> float:
    """Exponential backoff; take MAX of computed delay and any
    server-suggested wait so we always honor a 429's 'try again in N'.
    """
    computed = _BACKOFF_BASE * (_BACKOFF_EXP**attempt)
    return max(computed, suggested)


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
        log.info(
            "🌱 first run — backfilling %s → %s",
            start.date().isoformat(),
            end.date().isoformat(),
        )
    else:
        start = latest + timedelta(hours=1)
        log.info(
            "🔄 incremental pull from %s → %s",
            start.isoformat(),
            end.isoformat(),
        )
    if start >= end:
        log.info("✅ already current — no new rows to pull")
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
    """Page through gridstatus REST returning a stitched polars DataFrame.

    Paces requests at 1.1s/each to stay clear of the free-tier 1-req/sec
    limit. On 429, sleeps the gridstatus-suggested wait then retries.
    """
    headers = {"x-api-key": os.environ["GRIDSTATUS_API_KEY"]}
    url = f"{_GRIDSTATUS_BASE}/datasets/{dataset}/query"
    frames: list[pl.DataFrame] = []
    cursor: str | None = None
    page = 0
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        while True:
            if page > 0:
                time.sleep(_PACE_SECONDS)
            page += 1
            params: dict[str, Any] = {
                "start_time": start.strftime("%Y-%m-%dT%H:%M:%S"),
                "end_time": end.strftime("%Y-%m-%dT%H:%M:%S"),
                "filter_column": filter_column,
                "filter_value": filter_value,
                "limit": _PAGE_SIZE,
                # Reason: JSON (not CSV) so we can read `meta.cursor` from
                # the response body — CSV format omits cursor info entirely
                # and silently terminates pagination after one page.
                # array-of-arrays keeps payload compact (column names once
                # in meta, rows as bare arrays).
                "return_format": "json",
                "json_schema": "array-of-arrays",
            }
            if cursor:
                params["cursor"] = cursor
            log.info(
                "📡 gridstatus page %d (cursor=%s)", page, "yes" if cursor else "first"
            )
            resp = _request_with_429_retry(client, url, params, headers)
            payload = resp.json()
            meta = payload.get("meta", {})
            cursor = meta.get("cursor") if meta.get("hasNextPage") else None
            # array-of-arrays shape: data[0] = column header row,
            # data[1:] = actual data rows. NOT meta.columns.
            data = payload.get("data", [])
            if len(data) > 1:
                cols, rows = data[0], data[1:]
                page_df = pl.DataFrame(rows, schema=cols, orient="row")
                log.info("📥 page %d: %d rows", page, page_df.height)
                frames.append(page_df)
            if not cursor:
                break
    total = sum(f.height for f in frames)
    log.info("📦 fetched %d rows across %d page(s)", total, page)
    if not frames:
        return pl.DataFrame(
            schema={"ts": pl.Datetime("us", "UTC"), "value": pl.Float64}
        )
    return pl.concat(frames)


def _request_with_429_retry(
    client: httpx.Client,
    url: str,
    params: dict[str, Any],
    headers: dict[str, str],
) -> httpx.Response:
    """GET with exponential-backoff retry on 429, 5xx, and network errors.

    Mirrors gridstatusio SDK defaults: max 5 retries, base 2.0s, exp 2.0,
    so worst case backs off 2 + 4 + 8 + 16 + 32 = 62s before giving up.
    Honors 429 'try again in N seconds' as a floor on the next sleep.
    """
    last_exc: Exception | None = None
    last_resp: httpx.Response | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = client.get(url, params=params, headers=headers)
        except (httpx.NetworkError, httpx.TimeoutException) as e:
            last_exc = e
            wait = _backoff_wait(attempt)
            log.warning(
                "gridstatus network error %s attempt %d/%d, sleeping %.1fs",
                type(e).__name__,
                attempt + 1,
                _MAX_RETRIES + 1,
                wait,
            )
            time.sleep(wait)
            continue
        last_resp = resp
        if resp.status_code == 429:
            suggested = _parse_retry_seconds(resp.text)
            wait = _backoff_wait(attempt, suggested=suggested)
            log.warning(
                "gridstatus 429 attempt %d/%d, sleeping %.1fs",
                attempt + 1,
                _MAX_RETRIES + 1,
                wait,
            )
            time.sleep(wait)
            continue
        if resp.status_code in _RETRY_STATUS:
            wait = _backoff_wait(attempt)
            log.warning(
                "gridstatus %d attempt %d/%d, sleeping %.1fs",
                resp.status_code,
                attempt + 1,
                _MAX_RETRIES + 1,
                wait,
            )
            time.sleep(wait)
            continue
        # Non-retriable response (or 2xx) — return/raise immediately.
        resp.raise_for_status()
        return resp
    # Exhausted retries.
    if last_resp is not None:
        last_resp.raise_for_status()
        return last_resp
    assert last_exc is not None
    raise last_exc


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
    log.info("💾 loaded %d rows into timeseries_data", df.height)


def process(config: Config) -> None:
    """Execute ETL: extract → transform → load."""
    log.info(
        "🌐 ETL start — dataset=%s location=%s",
        config.gridstatus_dataset,
        config.settlement_point,
    )
    raw = extract(config)
    clean = transform(raw)
    if not clean.is_empty():
        log.info(
            "🧹 transformed: %d rows, ts range %s → %s",
            clean.height,
            clean["ts"].min(),
            clean["ts"].max(),
        )
    load(clean, config)
    log.info("✅ ETL done")
