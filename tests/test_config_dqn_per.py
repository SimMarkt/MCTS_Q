"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

test_config_dqn:
> Test cases for the Deep Q-Network model in mctsq_config_dqn_per.py
---------------------------------------------------------------------------------------------
"""

import pytest

def test_dqnmodel_instantiation():
    """Test instantiation of the DQNModel class."""
    from src.mctsq_config_dqn_per import DQNModel
    model = DQNModel(
        el_input_dim=1,
        process_input_dim=6,
        gas_eua_input_dim=2,
        action_dim=5,
        embed_dim=8,
        hidden_layers=1,
        hidden_units=16,
        buffer_capacity=100,
        batch_size=8,
        gamma=0.99,
        lr=0.001,
        price_encoder_type="gru",
        process_encoder_type="gru",
        gas_eua_encoder_type="gru",
        activation="relu",
        learning_starts=1,
        seed=42
    )
    assert hasattr(model, "policy_net")
    assert hasattr(model, "target_net")
    assert hasattr(model, "replay_buffer")

def test_triple_encoder_dqn_forward():
    """Test forward pass of the TripleEncoderDQN model."""
    import torch
    from src.mctsq_config_dqn_per import TripleEncoderDQN

    batch_size = 4
    seq_length = 10
    el_input_dim = 1
    process_input_dim = 6
    gas_eua_input_dim = 2
    embed_dim = 8
    hidden_layers = 1
    hidden_units = 16
    action_dim = 5

    model = TripleEncoderDQN(
        el_input_dim, process_input_dim, gas_eua_input_dim,
        embed_dim, hidden_layers, hidden_units, action_dim,
        price_encoder_type="gru", process_encoder_type="gru", gas_eua_encoder_type="gru"
    )

    price_data = torch.randn(batch_size, seq_length, el_input_dim)
    process_data = torch.randn(batch_size, seq_length, process_input_dim)
    gas_eua_data = torch.randn(batch_size, seq_length, gas_eua_input_dim)

    out = model(price_data, process_data, gas_eua_data)
    assert out.shape == (batch_size, action_dim)

def test_prioritized_replay_buffer_sample():
    import numpy as np
    from src.mctsq_config_dqn_per import PrioritizedReplayBuffer

    buffer = PrioritizedReplayBuffer(capacity=10)
    for i in range(10):
        buffer.push(
            state={"Elec_Price": 1, "T_CAT": 2, "H2_in_MolarFlow": 3, "CH4_syn_MolarFlow": 4,
                   "H2_res_MolarFlow": 5, "H2O_DE_MassFlow": 6, "Elec_Heating": 7,
                   "Gas_Price": 8, "EUA_Price": 9},
            action=i % 5,
            reward=float(i),
            next_state={"Elec_Price": 1, "T_CAT": 2, "H2_in_MolarFlow": 3, "CH4_syn_MolarFlow": 4,
                        "H2_res_MolarFlow": 5, "H2O_DE_MassFlow": 6, "Elec_Heating": 7,
                        "Gas_Price": 8, "EUA_Price": 9},
            done=False
        )
    batch = buffer.sample(batch_size=4)
    assert len(batch) == 7  # states, actions, rewards, next_states, dones, weights, indices
    assert batch[0].shape[0] == 4  # states