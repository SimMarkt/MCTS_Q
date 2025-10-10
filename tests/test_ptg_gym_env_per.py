"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

test_ptg_gym_env_per:
> Test cases for the RL environment including proper reward calculation and time step handling 
  in ptg_gym_env.py.
---------------------------------------------------------------------------------------------
"""

import pytest
import numpy as np
from env.ptg_gym_env import PTGEnv

@pytest.fixture
def minimal_env_kwargs():
    """Provides minimal environment keyword arguments for testing."""
    dummy_op = np.zeros((10, 7))
    meth_stats_load = {
        "Meth_H2_flow": [0.0, 0.00701, 0.0198],
        "Meth_CH4_flow": [0.0, 0.00172, 0.0048],
        "Meth_H2_res_flow": [0.0, 0.000054, 0.000151],
        "Meth_H2O_flow": [0.0, 0.0624, 0.458545],
    }
    return {
        "parallel": "Singleprocessing",
        "eps_ind": np.arange(10),
        "sim_step": 600,
        "time_step_op": 2,
        "ptg_standby": 0,
        "ptg_cooldown": 1,
        "ptg_startup": 2,
        "ptg_partial_load": 3,
        "ptg_full_load": 4,
        "e_r_b": np.zeros((3, 13, 100)),
        "g_e": np.zeros((2, 2, 100)),
        "act_ep_h": 0,
        "act_ep_d": 0,
        "scenario": 2,
        "raw_modified": "raw",
        "operation": "OP1",
        "meth_stats_load": meth_stats_load,
        "el_l_b": -10, "el_u_b": 90,
        "gas_l_b": 0.4, "gas_u_b": 32,
        "eua_l_b": 23, "eua_u_b": 98,
        "T_l_b": 10, "T_u_b": 600,
        "h2_l_b": 0, "h2_u_b": 100,
        "ch4_l_b": 0, "ch4_u_b": 50,
        "h2_res_l_b": 0, "h2_res_u_b": 10,
        "h2o_l_b": 0, "h2o_u_b": 20,
        "heat_l_b": 0, "heat_u_b": 1800,
        "H_u_CH4": 35.883,
        "H_u_H2": 10.783,
        "h_H2O_evap": 2257,
        "eta_CHP": 0.38,
        "eeg_el_price": 17.84,
        "dt_water": 90,
        "cp_water": 4.18,
        "ch4_price_fix": 15.0,
        "heat_price": 4.6,
        "o2_price": 10.2,
        "water_price": 6.4,
        "Molar_mass_CO2": 44.01,
        "min_load_electrolyzer": 0.032,
        "Molar_mass_H2O": 18.02,
        "rho_water": 998,
        "Meth_State": [2, 5, 5],
        "Meth_Action": [6, 10, 11],
        "Meth_Hot_Cold": [0, 1, 1],
        "Meth_T_cat": [11.0, 451.0, 451.0],
        "Meth_el_heating": [0.0, 231.0, 350.0],
        "max_h2_volumeflow": 0.0198,
        "eps_len_d": 1,
        "convert_mol_to_Nm3": 0.025,
        "n_eps_loops": 1,
        "standby_down": dummy_op,
        "standby_up": dummy_op,
        "cooldown": dummy_op,
        "startup_cold": dummy_op,
        "startup_hot": dummy_op,
        "op1_start_p": dummy_op,
        "op2_start_f": dummy_op,
        "op3_p_f": dummy_op,
        "op4_p_f_p_5": dummy_op,
        "op5_p_f_p_10": dummy_op,
        "op6_p_f_p_15": dummy_op,
        "op7_p_f_p_22": dummy_op,
        "op8_f_p": dummy_op,
        "op9_f_p_f_5": dummy_op,
        "op10_f_p_f_10": dummy_op,
        "op11_f_p_f_15": dummy_op,
        "op12_f_p_f_20": dummy_op,
        "reward_level": [0],
        "noise": 0.0,
        "np_random": np.random.default_rng(),
        "t_cat_standby": 0,
        "time2_start_f_p": 0,
        "time1_p_f_p": 0,
        "time2_p_f_p": 0,
        "time_p_f": 0,
        "time34_p_f_p": 0,
        "time45_p_f_p": 0,
        "time5_p_f_p": 0,
        "i_fully_developed": 0,
        "j_fully_developed": 0,
        "time1_start_p_f": 0,
        "time1_f_p_f": 0,
        "time_f_p": 0,
        "time23_f_p_f": 0,
        "time34_f_p_f": 0,
        "time45_f_p_f": 0,
        "time5_f_p_f": 0,
        "rew_l_b": 0,
        "rew_u_b": 1,
        "price_ahead": 13,
        "action_type": "discrete",
        "time_step_size_sim": 600,
        "meth_ch4_flow": 0.0048,
        "meth_h2_flow": 0.0198,
        "meth_h2_res_flow": 0.000151,
        "meth_h2o_flow": 0.458545,
        "meth_el_heating": 350.0,
        "meth_state": 2,
        "state_change_penalty": 0.0,
        "r_0": 0.0,
        "state_change": False,
        "g_e_act": np.array([[15.0, 0.0], [98.0, 0.0]]),  # [gas_price, eua_price]
        "e_r_b_act": np.array([[10.0]*13, [0.0]*13, [0.0]*13]),  # [el_price, pot_rew, part_full_b]
        "seq_length": 10,
        "price_past" : 12
    }


def test_ptg_env_reward_calculation(monkeypatch, minimal_env_kwargs, request):
    """Test reward calculation for all values in a sinusoidal electricity price pattern."""
    import env.ptg_gym_env_per
    env.ptg_gym_env_per.ep_index = 0

    rewards = []

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

    env = PTGEnv(dict_input=minimal_env_kwargs)
    obs, info = env.reset()
    env.meth_ch4_flow = minimal_env_kwargs["meth_ch4_flow"]
    env.meth_h2_flow = minimal_env_kwargs["meth_h2_flow"]
    env.meth_h2_res_flow = minimal_env_kwargs["meth_h2_res_flow"]
    env.meth_h2o_flow = minimal_env_kwargs["meth_h2o_flow"]
    env.meth_el_heating = minimal_env_kwargs["meth_el_heating"]

    for i in [5, 50]:
        env.e_r_b_act[0, 0] = el_price[i]
        if i == 5:
            env.g_e_act[0, 0] = gas_price[0]
            env.g_e_act[1, 0] = eua_price[0]
        else:  # i == -10
            env.g_e_act[0, 0] = gas_price[1]
            env.g_e_act[1, 0] = eua_price[1]

        reward = env._get_reward()
        rewards.append(reward)
        print(f"Reward: {reward}")

        assert isinstance(reward, float)

    assert round(rewards[0], 8) == round(25.71902518, 8)
    assert round(rewards[1], 8) == round(2.826585172281934, 8)
