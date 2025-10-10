"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

test_utils:
> Test cases for utility functions including data import in mctsq_utils.py.
---------------------------------------------------------------------------------------------
"""

import pytest
import numpy as np
import pandas as pd
from src.mctsq_utils import import_market_data, import_data

def test_import_market_data(tmp_path):
    """Test importing market data from CSV file."""
    df = pd.DataFrame({
        "Time": ["01-01-2020 00:00", "01-01-2020 01:00"],
        "Day-Ahead-price [Euro/MWh]": [10, 20],
        "Gas price [Euro/MWh]": [5, 6],
        "EUA price [Euro/tCO2]": [1, 2]
    })
    csv_path = tmp_path / "market.csv"
    df.to_csv(csv_path, sep=";", index=False)
    arr = import_market_data(str(csv_path.name), "el", str(tmp_path))
    assert isinstance(arr, np.ndarray)

def test_import_data(tmp_path):
    """Test importing data from CSV file."""
    df = pd.DataFrame({
        "Time [s]": [0, 3600],
        "T_cat [gradC]": [250, 255],
        "n_h2 [mol/s]": [1.0, 1.1],
        "n_ch4 [mol/s]": [0.5, 0.6],
        "n_h2_res [mol/s]": [0.1, 0.2],
        "m_DE [kg/h]": [2.0, 2.1],
        "Pel [W]": [100, 110]
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, sep=";", index=False)
    arr = import_data(str(csv_path.name), str(tmp_path))
    assert isinstance(arr, np.ndarray)
