"""Utility functions for ML pipeline."""

import pandas as pd

from src.models import XGBOOST_FEATURE_COLUMNS


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
        elif col.startswith("value_lag_"):
            row[col] = 0.0
        else:
            row[col] = getattr(timestamp, col)
    return pd.DataFrame({k: [v] for k, v in row.items()})
