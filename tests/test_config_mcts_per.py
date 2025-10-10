"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

test_config_mcts_per:
> Test cases for the MCTS model in mctsq_config_mcts_per.py
---------------------------------------------------------------------------------------------
"""

import pytest

def test_mctsqconfiguration_load(monkeypatch, dummy_mctsq_config):
    """Test loading of MCTSQ configuration."""
    import builtins
    import yaml
    from src.mctsq_config_mcts_per import MCTSQConfiguration

    real_open = builtins.open
    def open_patch(file, *args, **kwargs):
        if file == "config/config_mctsq.yaml":
            return real_open(dummy_mctsq_config, *args, **kwargs)
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", open_patch)
    config = MCTSQConfiguration()
    assert config.iterations == 2
    assert config.maximum_depth == 5

@pytest.mark.parametrize("meth_state,expected_actions", [
    (0, [0, 1, 2]),      # standby
    (1, [0, 1, 2]),      # cooldown
    (2, [0, 1, 3, 4]),   # startup
    (3, [0, 1, 3, 4]),   # partial load
    (4, [0, 1, 3, 4]),   # full load
    (99, [0, 1, 2, 3, 4]) # unknown/default
])
def test_mctsnode_legal_actions(meth_state, expected_actions):
    """Test legal actions based on meth_state."""
    from src.mctsq_config_mcts_per import MCTSNode
    state_c = {'Meth_State': meth_state}
    node = MCTSNode(state_c)
    assert set(node.get_legal_actions()) == set(expected_actions)
