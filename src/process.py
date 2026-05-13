"""Data processing pipeline for fetching and storing timeseries data."""

import os
from datetime import UTC, datetime
from typing import TypedDict

import pandera.polars as pa
import polars as pl
import psycopg2
import requests

from src.config import Config
from src.models import TimeseriesData


class GenericApiResponse(TypedDict):
    """Generic API response structure."""

    prices: list[list[int | float]]


def extract(config: Config) -> list[list[int | float]]:
    """Extract data from CoinGecko API.

    Args:
        config: Configuration object

    Returns:
        List of [timestamp_ms, price] pairs
    """
    params = {"vs_currency": "usd", "days": "1", "interval": "daily"}
    response = requests.get(config.api_url, params=params, timeout=30)
    response.raise_for_status()
    data: GenericApiResponse = response.json()
    return data["prices"]


@pa.check_types
def transform(prices: list[list[int | float]]) -> TimeseriesData:
    """Transform API prices to timeseries DataFrame.

    Args:
        prices: List of [timestamp_ms, price] pairs from API

    Returns:
        Validated timeseries DataFrame
    """
    df = pl.DataFrame(
        {
            "ts": [datetime.fromtimestamp(ts / 1000, tz=UTC) for ts, _ in prices],
            "value": [price for _, price in prices],
        }
    )

    return TimeseriesData(df)


@pa.check_types
def load(df: TimeseriesData, _config: Config) -> None:
    """Load data into TimescaleDB with upsert.

    Args:
        df: Timeseries data to insert
        config: Configuration object
    """
    with psycopg2.connect(os.environ["TIMESERIES_URL"]) as conn:
        with conn.cursor() as cursor:
            # Insert with ON CONFLICT DO UPDATE
            for row in df.iter_rows(named=True):
                cursor.execute(
                    """
                    INSERT INTO timeseries_data (ts, value)
                    VALUES (%s, %s)
                    ON CONFLICT (ts) DO UPDATE SET value = EXCLUDED.value
                    """,
                    (row["ts"], row["value"]),
                )
        conn.commit()


def process(config: Config) -> None:
    """Execute ETL pipeline: extract, transform, load.

    Args:
        config: Configuration object
    """
    values = extract(config)
    df = transform(values)
    load(df, config)
