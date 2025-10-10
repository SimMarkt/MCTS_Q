"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

conftest:
> Pytest fixtures for unit tests.
---------------------------------------------------------------------------------------------
"""

import pytest
import yaml

@pytest.fixture
def dummy_train_config(tmp_path):
    """Provides a dummy training configuration file."""
    config = {
        "com_conf": "pc",
        "model_conf": "simple_train",
        "r_seed_train": [89],
        "r_seed_test": [531]
    }
    config_path = tmp_path / "config_train.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return str(config_path)


@pytest.fixture
def dummy_mctsq_config(tmp_path):
    """Provides a dummy MCTS_Q configuration file."""
    config = {
        "iterations": 2,
        "c_init": 1.0,
        "c_base": 10,
        "maximum_depth": 5,
        "learning_rate": 0.001,
        "discount_factor": 0.99,
        "hidden_layers": 1,
        "hidden_units": 16,
        "buffer_size": 100,
        "batch_size": 8,
        "activation": "relu",
        "learning_starts": 1,
        "target_update": 10,
        "tau": 0.005,
        "seq_length": 2,
        "seq_step": 1,
        "embed_dim": 8,
        "price_encoder_type": "gru",
        "process_encoder_type": "gru",
        "gas_eua_encoder_type": "gru"
    }
    config_path = tmp_path / "config_mctsq.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return str(config_path)
