"""Tests for data processing pipeline."""

from datetime import datetime, UTC

import polars as pl

from src.process import transform


def test_transform_converts_api_to_dataframe() -> None:
    """Test that transform converts API response to DataFrame.

    Happy path: Convert prices array to timeseries DataFrame.
    """
    # Arrange
    prices = [[1728086400000, 62103.01], [1728172800000, 62091.93]]
    expected_timestamp = datetime(2024, 10, 5, 0, 0, tzinfo=UTC)

    # Act
    actual = transform(prices)

    # Assert
    assert isinstance(actual, pl.DataFrame)
    assert len(actual) == 2
    assert "timestamp" in actual.columns
    assert "value" in actual.columns
    assert actual["value"][0] == 62103.01
    assert actual["timestamp"][0].replace(tzinfo=UTC) == expected_timestamp
