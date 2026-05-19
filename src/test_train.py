"""Unit tests for select_champion with 3 candidate models."""

from src.models import PredictiveModels
from src.train import select_champion


def test_chooses_lightgbm_when_lowest_first_run() -> None:
    """LightGBM has lowest MAE -> picked as first champion."""
    actual = select_champion(15000.0, 14000.0, 12000.0, current_champion=None)
    assert actual == PredictiveModels.LIGHTGBM


def test_chooses_prophet_when_lowest_first_run() -> None:
    """Prophet has lowest MAE -> picked as first champion."""
    actual = select_champion(11000.0, 15000.0, 14000.0, current_champion=None)
    assert actual == PredictiveModels.PROPHET


def test_chooses_xgboost_when_lowest_first_run() -> None:
    """XGBoost has lowest MAE -> picked as first champion."""
    actual = select_champion(15000.0, 11000.0, 14000.0, current_champion=None)
    assert actual == PredictiveModels.XGBOOST


def test_keeps_champion_when_better_than_all_challengers() -> None:
    """Prophet champion still has lowest MAE -> stays champion."""
    actual = select_champion(
        11000.0, 15000.0, 14000.0, current_champion=PredictiveModels.PROPHET
    )
    assert actual == PredictiveModels.PROPHET


def test_promotes_lightgbm_when_it_beats_current_champion() -> None:
    """LightGBM beats current Prophet champion -> promoted."""
    actual = select_champion(
        15000.0, 14000.0, 11000.0, current_champion=PredictiveModels.PROPHET
    )
    assert actual == PredictiveModels.LIGHTGBM


def test_keeps_champion_when_no_challenger_beats_it() -> None:
    """All challengers >= champion -> champion stays (baseline-vs-champion check is downstream)."""
    actual = select_champion(
        25000.0, 26000.0, 27000.0, current_champion=PredictiveModels.PROPHET
    )
    assert actual == PredictiveModels.PROPHET
