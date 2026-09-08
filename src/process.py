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
# NOTE: don't pass `limit` to the query — gridstatus treats it as a
# total-rows cap (not per-page). Letting it default = server-side
# page_size=50000 + cursor pagination walks the full window.
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
    filter_column: str | None,
    filter_value: str | None,
    start: datetime,
    end: datetime,
    publish_time_start: datetime | None = None,
) -> pl.DataFrame:
    """Page through gridstatus REST returning a stitched polars DataFrame.

    Paces requests at 1.1s/each to stay clear of the free-tier 1-req/sec
    limit. On 429, sleeps the gridstatus-suggested wait then retries.
    `filter_column` + `filter_value` are optional — drop them entirely
    for system-wide datasets (load forecast, etc).

    `publish_time_start` is a separate, independent filter (verified
    against gridstatus's REST reference — it filters `publish_time_column`,
    `start_time`/`end_time` still filter `time_index_column`) — pass it to
    get only rows published after a watermark instead of re-walking a
    full time-index window. See `_sync_load_forecast_incremental`.
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
                # No `limit` — gridstatus treats it as a hard total cap.
                # JSON+array-of-arrays so we can read meta.cursor for
                # pagination (CSV omits cursor info entirely).
                "return_format": "json",
                "json_schema": "array-of-arrays",
            }
            if filter_column and filter_value:
                params["filter_column"] = filter_column
                params["filter_value"] = filter_value
            if publish_time_start:
                params["publish_time_start"] = publish_time_start.strftime(
                    "%Y-%m-%dT%H:%M:%S.%f"
                )
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
                # infer_schema_length=None → look at all rows. Without
                # this, a column whose first ~100 rows are null gets
                # inferred as Null type, then real floats below fail
                # to append (ComputeError on mixed-null float cols).
                page_df = pl.DataFrame(
                    rows, schema=cols, orient="row", infer_schema_length=None
                )
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
    """Upsert into timeseries_data via COPY + temp table.

    Row-by-row INSERT over a remote postgres = ~50ms RTT * N rows;
    38K rows took 15+ min without finishing. COPY streams in seconds,
    then a single MERGE-style INSERT...SELECT...ON CONFLICT applies the
    upsert. Total: a few seconds for tens of thousands of rows.
    """
    if df.is_empty():
        return
    csv_buf = io.StringIO()
    for row in df.iter_rows(named=True):
        # ISO-8601 with explicit UTC suffix — Postgres timestamptz parses cleanly.
        ts_str = row["ts"].isoformat()
        csv_buf.write(f"{ts_str}\t{row['value']}\n")
    csv_buf.seek(0)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE _ts_stage (ts TIMESTAMPTZ, value DOUBLE PRECISION) "
            "ON COMMIT DROP"
        )
        cur.copy_from(csv_buf, "_ts_stage", sep="\t", columns=("ts", "value"))
        # SELECT DISTINCT ON dedupes any ts that snuck through (e.g. DST
        # fall-back hour, or gridstatus revisions present in same window).
        # ON CONFLICT cannot update the same row twice in one statement,
        # so dedupe BEFORE the conflict clause sees the rows.
        cur.execute(
            "INSERT INTO timeseries_data (ts, value) "
            "SELECT DISTINCT ON (ts) ts, value FROM _ts_stage ORDER BY ts "
            "ON CONFLICT (ts) DO UPDATE SET value = EXCLUDED.value"
        )
        conn.commit()
    log.info("💾 loaded %d rows into timeseries_data via COPY", df.height)


_LOAD_FORECAST_DDL: Final[str] = """
    CREATE TABLE IF NOT EXISTS load_forecasts_raw (
        interval_start_utc TIMESTAMPTZ NOT NULL,
        publish_time_utc   TIMESTAMPTZ NOT NULL,
        load_mw            DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (interval_start_utc, publish_time_utc)
    )
"""


def _ensure_load_forecasts_table() -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(_LOAD_FORECAST_DDL)
        conn.commit()


def _latest_load_forecast_publish() -> datetime | None:
    """Most-recent publish_time_utc already stored (None if empty)."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT MAX(publish_time_utc) FROM load_forecasts_raw")
        row = cur.fetchone()
    return row[0] if row and row[0] else None


_LOAD_FORECAST_LOOKBACK_DAYS: Final[int] = 60
_LOAD_FORECAST_LOOKAHEAD_DAYS: Final[int] = 7
# gridstatus caps a single query at ~50k rows + hasNextPage=False.
# Load forecast publishes ~24 rows/hour (one publish/hour for each of
# 168 forecast horizons) → 50k / 24 ≈ 87 hours ≈ 3.5 days. Use a
# 3-day chunk to stay safely under the cap.
_LOAD_FORECAST_CHUNK_DAYS: Final[int] = 3


def _clean_load_forecast_chunk(raw: pl.DataFrame, zone_col: str) -> pl.DataFrame:
    """Project a raw gridstatus load-forecast page to (ts, publish, load_mw).

    Shared by the full chunked walk and the watermark sync — same source
    columns, same target shape either way.

    Raises:
        ValueError: if `zone_col` isn't a column in `raw` (bad config).
    """
    if raw.is_empty():
        return raw
    if zone_col not in raw.columns:
        raise ValueError(
            f"load_forecast_zone {zone_col!r} not in dataset columns {raw.columns!r}"
        )
    return (
        raw.select(
            pl.col("interval_start_utc")
            .str.to_datetime(time_zone="UTC")
            .alias("interval_start_utc"),
            pl.col("publish_time_utc")
            .str.to_datetime(time_zone="UTC")
            .alias("publish_time_utc"),
            pl.col(zone_col).cast(pl.Float64).alias("load_mw"),
        )
        .drop_nulls()
        .unique(subset=["interval_start_utc", "publish_time_utc"])
    )


def _process_load_forecast(config: Config) -> None:
    """Full chunked walk of ERCOT load forecast -> load_forecasts_raw.

    Scope: last 60 days + next 7 days. Pulled in 3-day chunks because
    gridstatus' load-forecast dataset hits the 50k-row page cap at
    ~3.5 days per query (no working cursor for this dataset).
    UPSERT on (interval_start_utc, publish_time_utc) is idempotent
    across chunk overlaps + re-runs.

    Used two ways (see `_sync_load_forecast`): as the one-time bootstrap
    when `load_forecasts_raw` is empty, and as the monthly full
    reconciliation — gridstatus's own guidance for forecast datasets that
    republish for the same interval is to pair a cheap daily watermark
    sync with an occasional full walk that catches late corrections the
    watermark can miss.

    LEAKAGE NOTE: training applies the strict `<= ts - 24h` filter at
    SELECT time in `train.load_timeseries_data`. Mid-day same-day
    publishes in this raw table are filtered there.
    """
    _ensure_load_forecasts_table()
    now = datetime.now(UTC)
    window_start = now - timedelta(days=_LOAD_FORECAST_LOOKBACK_DAYS)
    window_end = now + timedelta(days=_LOAD_FORECAST_LOOKAHEAD_DAYS)
    log.info(
        "🔄 load-forecast full walk %s -> %s (3-day chunks)",
        window_start.date(),
        window_end.date(),
    )
    total = 0
    chunk_start = window_start
    while chunk_start < window_end:
        chunk_end = min(
            chunk_start + timedelta(days=_LOAD_FORECAST_CHUNK_DAYS), window_end
        )
        try:
            raw = _fetch_dataset(
                dataset=config.load_forecast_dataset,
                filter_column=None,
                filter_value=None,
                start=chunk_start,
                end=chunk_end,
            )
        except httpx.HTTPStatusError as e:
            # 403 = daily quota; 4xx = client cap; bail rather than
            # block the rest of the pipeline. Training proceeds on
            # whatever forecast rows were loaded before the cap.
            log.warning(
                "load-forecast chunk %s -> %s failed: %s — skipping remainder",
                chunk_start.date(),
                chunk_end.date(),
                e,
            )
            break
        chunk_start = chunk_end
        clean = _clean_load_forecast_chunk(raw, config.load_forecast_zone)
        if clean.is_empty():
            continue
        _load_forecast_rows(clean)
        total += clean.height
    log.info("💾 loaded %d total load-forecast rows", total)


def _sync_load_forecast_incremental(config: Config, watermark: datetime) -> None:
    """One cheap call: only load-forecast rows published after `watermark`.

    gridstatus's documented recipe for this exact dataset (Backfill and
    Incrementally Sync a Dataset) — `publish_time_start` filters
    server-side, so this replaces the ~23-chunk full walk with a single
    request on every day that isn't a reconciliation day.
    """
    _ensure_load_forecasts_table()
    now = datetime.now(UTC)
    window_start = now - timedelta(days=_LOAD_FORECAST_LOOKBACK_DAYS)
    window_end = now + timedelta(days=_LOAD_FORECAST_LOOKAHEAD_DAYS)
    publish_time_start = watermark + timedelta(microseconds=1)
    log.info(
        "🔄 load-forecast watermark sync — publishes after %s",
        publish_time_start.isoformat(),
    )
    try:
        raw = _fetch_dataset(
            dataset=config.load_forecast_dataset,
            filter_column=None,
            filter_value=None,
            start=window_start,
            end=window_end,
            publish_time_start=publish_time_start,
        )
    except httpx.HTTPStatusError as e:
        log.warning("load-forecast watermark sync failed: %s — skipping this cycle", e)
        return
    clean = _clean_load_forecast_chunk(raw, config.load_forecast_zone)
    if clean.is_empty():
        log.info("✅ load-forecast up to date — no new publishes")
        return
    _load_forecast_rows(clean)
    log.info(
        "💾 loaded %d new/revised load-forecast rows (watermark sync)", clean.height
    )


def _should_run_full_reconciliation(watermark: datetime | None, now: datetime) -> bool:
    """True on first-ever sync (no watermark) or the 1st of the month.

    gridstatus recommends pairing a watermark sync with periodic full
    reconciliation to catch late corrections the watermark alone can
    miss. Anchored to day-of-month rather than a day-count interval so
    it doesn't drift — the `schedule` lib has no native monthly cadence.
    """
    return watermark is None or now.day == 1


def _sync_load_forecast(config: Config, now: datetime | None = None) -> None:
    """Daily driver: watermark sync, with a monthly full reconciliation.

    Args:
        config: pipeline config.
        now: injectable for tests; defaults to wall-clock UTC now.
    """
    watermark = _latest_load_forecast_publish()
    now = now or datetime.now(UTC)
    if _should_run_full_reconciliation(watermark, now):
        reason = "no prior sync" if watermark is None else "monthly reconciliation"
        log.info("🗓 full load-forecast walk (%s)", reason)
        _process_load_forecast(config)
        return
    # Reason: _should_run_full_reconciliation already returned True above
    # when watermark is None — reaching here means it's set. Narrows the
    # type for the incremental path, which needs a real watermark.
    assert watermark is not None
    _sync_load_forecast_incremental(config, watermark)


def _load_forecast_rows(df: pl.DataFrame) -> None:
    """COPY+temp+UPSERT — same pattern as the SPP `load` function."""
    buf = io.StringIO()
    for row in df.iter_rows(named=True):
        buf.write(
            f"{row['interval_start_utc'].isoformat()}\t"
            f"{row['publish_time_utc'].isoformat()}\t"
            f"{row['load_mw']}\n"
        )
    buf.seek(0)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE _lf_stage "
            "(interval_start_utc TIMESTAMPTZ, publish_time_utc TIMESTAMPTZ, "
            "load_mw DOUBLE PRECISION) ON COMMIT DROP"
        )
        cur.copy_from(
            buf,
            "_lf_stage",
            sep="\t",
            columns=("interval_start_utc", "publish_time_utc", "load_mw"),
        )
        cur.execute(
            "INSERT INTO load_forecasts_raw "
            "(interval_start_utc, publish_time_utc, load_mw) "
            "SELECT DISTINCT ON (interval_start_utc, publish_time_utc) "
            "  interval_start_utc, publish_time_utc, load_mw "
            "FROM _lf_stage "
            "ORDER BY interval_start_utc, publish_time_utc "
            "ON CONFLICT (interval_start_utc, publish_time_utc) "
            "DO UPDATE SET load_mw = EXCLUDED.load_mw"
        )
        conn.commit()


def process(config: Config) -> None:
    """Execute ETL: SPP target + load forecast feature → Postgres.

    SPP extract is wrapped in an HTTPStatusError guard mirroring the
    existing pattern in `_process_load_forecast`. Reason: on 403
    (quota exhausted) or 404 (bad dataset slug) we log-and-skip so the
    daily scheduler doesn't crash and the load-forecast half still runs.
    """
    log.info(
        "🌐 ETL start — dataset=%s location=%s",
        config.gridstatus_dataset,
        config.settlement_point,
    )
    try:
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
    except httpx.HTTPStatusError as e:
        log.warning(
            "SPP extract failed (%s) — skipping SPP ETL this cycle, "
            "continuing to load-forecast",
            e,
        )
    log.info(
        "🌐 load-forecast ETL — dataset=%s zone=%s",
        config.load_forecast_dataset,
        config.load_forecast_zone,
    )
    _sync_load_forecast(config)
    log.info("✅ ETL done")
