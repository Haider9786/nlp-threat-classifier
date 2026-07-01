"""Unit tests for config loading and logging utilities."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from threat_classifier.utils import load_config


def test_load_config_success():
    config = load_config("config.yaml")
    assert "data" in config
    assert "model" in config
    assert "logging" in config


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")


def test_load_config_model_fields():
    config = load_config("config.yaml")
    assert "max_features" in config["model"]
    assert "test_size" in config["model"]
    assert "cv_folds" in config["model"]