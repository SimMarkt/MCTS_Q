"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

test_config_train:
> Test cases for the TrainConfiguration class in mctsq_config_train.py.
---------------------------------------------------------------------------------------------
"""

# pylint: disable=no-member

import builtins

import pytest
import yaml

from src.mctsq_config_train import TrainConfiguration

real_open = builtins.open

def test_train_config_load(monkeypatch, dummy_train_config):
    """Test loading of training configuration."""
    def open_patch(file, *args, **kwargs):
        if file == "config/config_train.yaml":
            return real_open(dummy_train_config, *args, **kwargs)
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", open_patch)
    train_conf = TrainConfiguration()
    assert train_conf.com_conf == "pc"
    assert train_conf.model_conf == "simple_train"

def test_train_config_invalid_com_conf(monkeypatch, tmp_path):
    """Test handling of invalid computation configuration."""
    config = {
        "com_conf": "invalid",
        "model_conf": "simple_train",
        "r_seed_train": [89],
        "r_seed_test": [531]
    }
    config_path = tmp_path / "config_train.yaml"
    with real_open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    def open_patch(file, *args, **kwargs):
        if file == "config/config_train.yaml":
            return real_open(config_path, *args, **kwargs)
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", open_patch)
    with pytest.raises(Exception):
        TrainConfiguration()
