"""Seed historical timeseries data from S3 SQL dump into PostgreSQL.

This script handles the one-time backfill of historical data.
For ongoing incremental updates, see src/process.py.
"""

import os
import tempfile

import psycopg2
import requests

from src.config import load_config

SQL_DUMP_URL = "https://fsenergy.s3.amazonaws.com/datasets/template_timeseries.sql"


def download_sql_dump(url: str) -> str:
    """Download SQL dump from S3 to temporary file.

    Args:
        url: S3 URL of the SQL dump file

    Returns:
        Path to downloaded temporary file

    """
    print(f"=� Downloading SQL dump from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
        f.write(response.text)
        temp_path = f.name

    print(f" Downloaded {len(response.text):,} bytes to {temp_path}")
    return temp_path


def execute_sql_dump(sql_path: str, timeseries_url: str) -> None:
    """Execute SQL dump against PostgreSQL database.

    Args:
        sql_path: Path to SQL dump file
        timeseries_url: PostgreSQL connection string

    """
    print("=�  Connecting to timeseries database...")
    conn = psycopg2.connect(timeseries_url)
    cursor = conn.cursor()

    print(f"� Executing SQL dump from {sql_path}...")
    with open(sql_path) as f:
        sql = f.read()
        cursor.execute(sql)
        conn.commit()

    cursor.close()
    conn.close()
    print(" SQL dump executed successfully")


def seed() -> None:
    """Download and load historical timeseries data into PostgreSQL."""
    # Load configuration
    load_config()

    timeseries_url = os.environ.get("TIMESERIES_URL", "")
    if not timeseries_url:
        raise ValueError("TIMESERIES_URL environment variable is required")

    # Download SQL dump
    sql_path = download_sql_dump(SQL_DUMP_URL)

    try:
        # Execute SQL dump
        execute_sql_dump(sql_path=sql_path, timeseries_url=timeseries_url)
    finally:
        # Cleanup temporary file
        if os.path.exists(sql_path):
            os.unlink(sql_path)
            print(f">� Cleaned up temporary file {sql_path}")


if __name__ == "__main__":
    seed()
