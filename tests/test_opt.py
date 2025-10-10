"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

test_opt:
> Test cases for the calculation of the potential reward and load identifier in mctsq_opt.py.
---------------------------------------------------------------------------------------------
"""

import pytest
import numpy as np
from src.mctsq_opt import calculate_optimum

def test_calculate_optimum(monkeypatch):
    """Test calculation of optimum reward and load identifier."""
    class DummyEnvConf:
        """Dummy environment configuration for testing (mock)."""
        scenario = 1
        meth_stats_load = {
            "Meth_H2_flow": [0.0, 0.00701, 0.0198],
            "Meth_CH4_flow": [0.0, 0.00172, 0.0048],
            "Meth_H2_res_flow": [0.0, 0.000054, 0.000151],
            "Meth_H2O_flow": [0.0, 0.0624, 0.458545],
            "Meth_State": [2, 5, 5],
            "Meth_Action": [6, 10, 11],
            "Meth_Hot_Cold": [0, 1, 1],
            "Meth_T_cat": [11.0, 451.0, 451.0],
            "Meth_el_heating": [0.0, 231.0, 350.0]
        }
        price_ahead = 13
        convert_mol_to_Nm3 = 0.02241407
        H_u_CH4 = 35.883
        H_u_H2 = 10.783
        h_H2O_evap = 2257
        eta_CHP = 0.38
        eeg_el_price = 17.84
        dt_water = 90
        cp_water = 4.18
        ch4_price_fix = 15.0
        heat_price = 4.6
        o2_price = 10.2
        water_price = 6.4
        Molar_mass_CO2 = 44.01
        min_load_electrolyzer = 0.032
        Molar_mass_H2O = 18.02
        rho_water = 998
        Meth_State = [2, 5, 5]
        Meth_Action = [6, 10, 11]
        Meth_Hot_Cold = [0, 1, 1]
        Meth_T_cat = [11.0, 451.0, 451.0]
        Meth_el_heating = [0.0, 231.0, 350.0]
        max_h2_volumeflow = convert_mol_to_Nm3 *  meth_stats_load['Meth_H2_flow'][2]

    monkeypatch.setattr("src.mctsq_opt.EnvConfiguration", lambda: DummyEnvConf)

    # Define dummy price data
    # Electricity price: sin wave, 3 days (3*24 points), scaled to [-10, 90]
    el_l_b, el_u_b = -10, 90
    el_len = 3 * 24
    el_price = (np.sin(np.linspace(0, 2 * np.pi, el_len)) + 1) / 2  # [0,1]
    el_price = el_l_b + (el_u_b - el_l_b) * el_price

    # Gas price: cos wave, 3 points, scaled to [0.4, 32]
    gas_l_b, gas_u_b = 0.4, 32
    gas_len = 3
    gas_price = (np.cos(np.linspace(0, 2 * np.pi, gas_len)) + 1) / 2  # [0,1]
    gas_price = gas_l_b + (gas_u_b - gas_l_b) * gas_price

    # EUA price: sin wave, 3 points, scaled to [23, 98]
    eua_l_b, eua_u_b = 23, 98
    eua_len = 3
    eua_price = (np.sin(np.linspace(0, 2 * np.pi, eua_len)) + 1) / 2  # [0,1]
    eua_price = eua_l_b + (eua_u_b - eua_l_b) * eua_price

    stats_names = [
        'steps_stats', 'el_price_stats', 'gas_price_stats', 'eua_price_stats', 
        'Meth_State_stats', 'Meth_Action_stats', 'Meth_Hot_Cold_stats', 'Meth_T_cat_stats', 
        'Meth_H2_flow_stats', 'Meth_CH4_flow_stats', 'Meth_H2O_flow_stats', 
        'Meth_el_heating_stats', 'Meth_ch4_revenues_stats', 'Meth_steam_revenues_stats', 
        'Meth_o2_revenues_stats', 'Meth_eua_revenues_stats', 'Meth_chp_revenues_stats', 
        'Meth_elec_costs_heating_stats', 'Meth_elec_costs_electrolyzer_stats', 
        'Meth_water_costs_stats', 'Meth_reward_stats', 'Meth_cum_reward_stats', 
        'pot_reward_stats', 'part_full_stats'
    ]
    result = calculate_optimum(el_price, gas_price, eua_price, "test", stats_names)
    assert isinstance(result, dict)
    assert all(k in result for k in stats_names)    # Test keys
    assert result['Meth_State_stats'][5] == 2
    assert result['Meth_State_stats'][-10] == 5
    assert round(result['Meth_ch4_revenues_stats'][5], 8) == 0
    assert round(result['Meth_ch4_revenues_stats'][-10], 8) == round(124.70588425, 8)
    assert round(result['Meth_steam_revenues_stats'][5], 8) == 0
    assert round(result['Meth_steam_revenues_stats'][-10], 8) == round(1.54284089, 8)
    assert round(result['Meth_o2_revenues_stats'][5], 8) == 0
    assert round(result['Meth_o2_revenues_stats'][-10], 8) == round(8.14814204, 8)
    assert round(result['Meth_eua_revenues_stats'][5], 8) == 0
    assert round(result['Meth_eua_revenues_stats'][-10], 8) == round(4.60098144, 8)
    assert round(result['Meth_elec_costs_heating_stats'][5], 8) == 0
    assert round(result['Meth_elec_costs_heating_stats'][-10], 8) == round(-1.48950662, 8)
    assert round(result['Meth_elec_costs_electrolyzer_stats'][5], 8) == 0
    assert round(result['Meth_elec_costs_electrolyzer_stats'][-10], 8) == round(-40.78487357, 8)
    assert round(result['Meth_water_costs_stats'][5], 8) == 0
    assert round(result['Meth_water_costs_stats'][-10], 8) == round(-0.01117762, 8)
    assert round(result['Meth_reward_stats'][5], 8) == round(-143.421015, 8)
    assert round(result['Meth_reward_stats'][-10], 8) == round(96.7122908, 8)
    assert round(result['Meth_cum_reward_stats'][-10], 8) == round(3016.75981673, 8)
