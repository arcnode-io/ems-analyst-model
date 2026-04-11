"""Tests for model training pipeline."""

from src.models import PredictiveModels
from src.train import select_champion


def test_select_champion_chooses_prophet_when_lower_mae() -> None:
    """Test that select_champion picks Prophet when it has lower MAE.

    Happy path: Prophet has lower MAE, should be selected as champion.
    """
    # Arrange
    prophet_mae = 12000.0
    xgboost_mae = 15000.0
    baseline_mae = 20000.0

    # Act
    actual = select_champion(
        prophet_mae, xgboost_mae, baseline_mae, current_champion=None
    )

    # Assert
    assert actual == PredictiveModels.PROPHET


def test_select_champion_chooses_xgboost_when_lower_mae() -> None:
    """Test that select_champion picks XGBoost when it has lower MAE.

    Edge case: XGBoost has lower MAE, should be selected as champion.
    """
    # Arrange
    prophet_mae = 15000.0
    xgboost_mae = 12000.0
    baseline_mae = 20000.0

    # Act
    actual = select_champion(
        prophet_mae, xgboost_mae, baseline_mae, current_champion=None
    )

    # Assert
    assert actual == PredictiveModels.XGBOOST


def test_select_champion_keeps_champion_when_better_than_challenger() -> None:
    """Test champion/challenger pattern keeps champion when it's better.

    Edge case: Current champion is better than new challenger.
    """
    # Arrange
    prophet_mae = 12000.0
    xgboost_mae = 15000.0
    baseline_mae = 20000.0
    current_champion = PredictiveModels.PROPHET

    # Act - retrain and Prophet is still best
    actual = select_champion(prophet_mae, xgboost_mae, baseline_mae, current_champion)

    # Assert
    assert actual == PredictiveModels.PROPHET


def test_select_champion_deploys_challenger_when_better() -> None:
    """Test champion/challenger pattern deploys challenger when it's better.

    Happy path: Challenger beats current champion.
    """
    # Arrange - XGBoost is now better than current Prophet champion
    prophet_mae = 15000.0
    xgboost_mae = 12000.0
    baseline_mae = 20000.0
    current_champion = PredictiveModels.PROPHET

    # Act
    actual = select_champion(prophet_mae, xgboost_mae, baseline_mae, current_champion)

    # Assert
    assert actual == PredictiveModels.XGBOOST


def test_select_champion_keeps_champion_when_baseline_is_better() -> None:
    """Test that champion is kept even when baseline is better (degradation alert).

    Failure case: System degrading, baseline beats champion.
    Should keep champion and alert (not regress to baseline).
    """
    # Arrange - baseline is better than champion (system degrading!)
    prophet_mae = 25000.0
    xgboost_mae = 26000.0
    baseline_mae = 20000.0
    current_champion = PredictiveModels.PROPHET

    # Act
    actual = select_champion(prophet_mae, xgboost_mae, baseline_mae, current_champion)

    # Assert - should keep champion, not regress to baseline
    assert actual == PredictiveModels.PROPHET
