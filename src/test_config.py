"""Loader tests — deploy-time customer-merge over baked cfg.yml."""

import os
from pathlib import Path
from unittest.mock import patch

from src.config import YamlScalar, _deep_merge, load_config


def test_load_config_returns_baked_defaults_without_customer_path_set() -> None:
    """No CFG_CUSTOMER_PATH set -> local stage in cfg.yml, untouched."""
    # Arrange
    env = {k: v for k, v in os.environ.items() if k != "CFG_CUSTOMER_PATH"}
    env["ENV"] = "local"
    with patch.dict(os.environ, env, clear=True):
        # Act
        cfg = load_config()

    # Assert
    assert cfg.mlflow_tracking_uri == "http://localhost:5000"
    assert cfg.site_id == "local"


def test_load_config_customer_merges_mlflow_tracking_uri(tmp_path: Path) -> None:
    """cfg file at CFG_CUSTOMER_PATH wins over baked defaults — the blocker field."""
    # Arrange
    customer = tmp_path / "analyst-cfg.yml"
    customer.write_text(
        "mlflow_tracking_uri: http://mlflow:5000\nsite_id: brookside_dc_1\n"
    )
    with patch.dict(
        os.environ, {"ENV": "local", "CFG_CUSTOMER_PATH": str(customer)}, clear=False
    ):
        # Act
        cfg = load_config()

    # Assert — overridden fields win
    assert cfg.mlflow_tracking_uri == "http://mlflow:5000"
    assert cfg.site_id == "brookside_dc_1"
    # Untouched fields keep the baked default
    assert cfg.settlement_point == "HB_NORTH"


def test_load_config_ignores_missing_customer_file(tmp_path: Path) -> None:
    """CFG_CUSTOMER_PATH pointing at a nonexistent file = no merge, no error."""
    # Arrange
    missing = tmp_path / "does-not-exist.yml"
    with patch.dict(
        os.environ, {"ENV": "local", "CFG_CUSTOMER_PATH": str(missing)}, clear=False
    ):
        # Act
        cfg = load_config()

    # Assert
    assert cfg.mlflow_tracking_uri == "http://localhost:5000"


def test_deep_merge_overlay_wins_per_key() -> None:
    """Sanity check — overlay's keys win, base's untouched keys survive."""
    # Arrange
    base: dict[str, YamlScalar] = {"a": 1, "b": "base"}
    overlay: dict[str, YamlScalar] = {"b": "customer", "c": True}

    # Act
    merged = _deep_merge(base, overlay)

    # Assert
    assert merged == {"a": 1, "b": "customer", "c": True}
