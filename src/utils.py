"""Utility functions for ML pipeline."""

import pandas as pd

from src.models import XGBOOST_FEATURE_COLUMNS


def create_xgboost_input_example(timestamp: pd.Timestamp) -> pd.DataFrame:
    """Create XGBoost input example from timestamp.

    Args:
        timestamp: Input timestamp

    Returns:
        DataFrame with time-based features for XGBoost
    """
    return pd.DataFrame(
        {
            col: [
                (
                    getattr(timestamp, col)
                    if col != "weekofyear"
                    else timestamp.isocalendar().week
                )
            ]
            for col in XGBOOST_FEATURE_COLUMNS
        }
    )
