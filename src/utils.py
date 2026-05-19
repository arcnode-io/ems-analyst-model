"""Utility functions for ML pipeline."""

from typing import Final

import pandas as pd

from src.models import XGBOOST_FEATURE_COLUMNS

# Columns that map directly to pandas.Timestamp attribute access.
_TIME_ATTR_COLUMNS: Final[frozenset[str]] = frozenset(
    {"hour", "day", "month", "year", "dayofweek", "dayofyear"}
)


def create_xgboost_input_example(timestamp: pd.Timestamp) -> pd.DataFrame:
    """One-row input example used by MLflow for signature inference.

    Time features come from the timestamp; lag values are zero
    placeholders — the real prediction-time values come from the
    history lookup in score._score_tree. MLflow just needs the schema
    to match (column names + dtypes).
    """
    row: dict[str, object] = {}
    for col in XGBOOST_FEATURE_COLUMNS:
        if col == "weekofyear":
            row[col] = int(timestamp.isocalendar().week)
        elif col in _TIME_ATTR_COLUMNS:
            row[col] = getattr(timestamp, col)
        else:
            # Lag + external-feature placeholders; the real values come
            # from history / load-forecast lookup in score._score_tree.
            row[col] = 0.0
    return pd.DataFrame({k: [v] for k, v in row.items()})
