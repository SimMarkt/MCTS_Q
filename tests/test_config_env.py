"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

test_config_env:
> Test cases for the EnvConfiguration class in mctsq_config_env.py.
---------------------------------------------------------------------------------------------
"""

# pylint: disable=no-member

import builtins

import pytest
import yaml

from src.mctsq_config_env import EnvConfiguration

real_open = builtins.open

def test_env_config_load(monkeypatch, tmp_path):
    """Test loading of environment configuration."""
    # Test datafile dict with dynamically generated keys
    datafile = {f"datafile_path{i}": f"file{i}.csv" for i in range(2, 19)}
    config = {
        "scenario": 1,
        "raw_modified": "raw",
        "operation": "OP1",
        "datafile_path": {
            "path": str(tmp_path) + "/",
            "datafile": datafile
        },
        "meth_stats_load": {
            "OP1": {
                "Meth_H2_flow": [0, 0.024, 0.048],
                "Meth_CH4_flow": [0, 0.008, 0.016],
                "Meth_H2_res_flow": [0, 0.001, 0.002],
                "Meth_H2O_flow": [0, 0.5, 1.3]
            }
        },
        "convert_mol_to_Nm3": 0.025
    }
    config_path = tmp_path / "config_env.yaml"
    with real_open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    def open_patch(file, *args, **kwargs):
        if file == "config/config_env.yaml":
            return real_open(config_path, *args, **kwargs)
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", open_patch)
    env_conf = EnvConfiguration()
    assert env_conf.scenario == 1
    assert env_conf.operation == "OP1"
    assert env_conf.convert_mol_to_Nm3 == 0.025

def test_env_config_invalid_operation(monkeypatch, tmp_path):
    """Test ValueError is raised for invalid operation."""
    datafile = {f"datafile_path{i}": f"file{i}.csv" for i in range(2, 19)}
    config = {
        "scenario": 1,
        "raw_modified": "raw",
        "operation": "INVALID",  # Invalid value
        "datafile_path": {
            "path": str(tmp_path) + "/",
            "datafile": datafile
        },
        "meth_stats_load": {
            "OP1": {
                "Meth_H2_flow": [0, 0.024, 0.048],
                "Meth_CH4_flow": [0, 0.008, 0.016],
                "Meth_H2_res_flow": [0, 0.001, 0.002],
                "Meth_H2O_flow": [0, 0.5, 1.3]
            }
        },
        "convert_mol_to_Nm3": 0.025
    }
    config_path = tmp_path / "config_env.yaml"
    with real_open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    def open_patch(file, *args, **kwargs):
        if file == "config/config_env.yaml":
            return real_open(config_path, *args, **kwargs)
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", open_patch)
    with pytest.raises(ValueError, match="Invalid load level specified"):
        EnvConfiguration()
