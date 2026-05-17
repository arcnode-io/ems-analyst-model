"""Unit tests for the gridstatus → timeseries_data transform."""

import polars as pl

from src.process import transform


def test_transform_picks_ts_and_spp_from_gridstatus_shape() -> None:
    """Transform projects gridstatus CSV rows to (ts, value) and dedupes."""
    # Arrange — pl.read_csv yields String columns; transform parses to datetime
    raw = pl.DataFrame(
        {
            "interval_start_utc": [
                "2026-05-17T00:00:00",
                "2026-05-17T01:00:00",
                "2026-05-17T00:00:00",  # duplicate — transform should dedupe
            ],
            "location": ["HB_NORTH", "HB_NORTH", "HB_NORTH"],
            "spp": [42.5, -12.3, 42.5],  # negative LMP is valid (oversupply)
        }
    )

    # Act
    actual = transform(raw)

    # Assert
    assert isinstance(actual, pl.DataFrame)
    assert len(actual) == 2  # dedup'd
    assert actual.columns == ["ts", "value"]
    assert actual["value"][0] == 42.5
    assert actual["value"][1] == -12.3  # negative price preserved


def test_transform_empty_input_returns_empty_df() -> None:
    """Empty extract (no new rows) round-trips to empty TimeseriesData."""
    # Arrange
    empty = pl.DataFrame(
        schema={
            "interval_start_utc": pl.Utf8,
            "location": pl.Utf8,
            "spp": pl.Float64,
        }
    )

    # Act
    actual = transform(empty)

    # Assert
    assert actual.is_empty()
    assert actual.columns == ["ts", "value"]
