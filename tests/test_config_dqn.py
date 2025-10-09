"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

test_config_dqn:
> Test cases for the Deep Q-Network model in mctsq_config_dqn.py
---------------------------------------------------------------------------------------------
"""

import pytest

def test_dqnmodel_instantiation():
    """Test instantiation of the DQNModel class."""
    from src.mctsq_config_dqn import DQNModel
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

def test_gru_attention_enc_output_shape():
    """Test output shape of the GRUAttentionEnc."""
    import torch
    from src.mctsq_config_dqn import GRUAttentionEnc
    batch_size, seq_length, input_dim, embed_dim = 4, 10, 6, 8
    enc = GRUAttentionEnc(input_dim, embed_dim)
    x = torch.randn(batch_size, seq_length, input_dim)
    out = enc(x)
    assert out.shape == (batch_size, embed_dim)
