# Test for memory-based simulation model of 20kW-Methanation

# ----------------------------------------------------------------------------------------------------------------------
print("Import libraries...")
import pandas as pd
import os
import time
from tqdm import tqdm
import logging
import math
import matplotlib.pyplot as plt

import numpy as np
# import gym
# from collections import OrderedDict
# from stable_baselines3.common.env_checker import check_env
# from gym.wrappers.time_limit import TimeLimit
# from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines.common.callbacks import CheckpointCallback

# Libraries with utility functions and classes
from src.mctsq_utils import load_data, initial_print, config_print, Preprocessing, create_envs, plot_results
from src.mctsq_config_env import EnvConfiguration
from src.mctsq_config_train import TrainConfiguration
from src.mctsq_config_mcts_V2 import MCTSQConfiguration, MCTS_Q

# from stable_baselines3 import PPO

# Import modules and utils
from utils import import_historic_prices, import_data, df_to_nparray

# ---------------------------------------------------------------------------------------------------------------------
print("Specify model parameters and classes...")  ### ToDO: PROVE VALUES BEFORE FINAL TRAINING AGAIN
class GlobalParams:
    def __init__(self):
        self.datafile_path1 = "data_5kW/data-day-ahead.csv"
        self.datafile_path2 = "data_5kW/data-meth_startup_cold.csv"
        self.datafile_path3 = "data_5kW/data-meth_startup_hot.csv"
        self.datafile_path4 = "data_5kW/data-meth_cooldown.csv"
        self.datafile_path5 = "data_5kW/data-meth_standby_down.csv"      # from operation to Hot-Standby
        self.datafile_path6 = "data_5kW/data-meth_standby_up.csv"      # from shutdown to Hot-Standby
        self.datafile_path7 = "data_5kW/data-meth_op1_start_p.csv"
        self.datafile_path8 = "data_5kW/data-meth_op2_start_f.csv"
        self.datafile_path9 = "data_5kW/data-meth_op3_p_f.csv"
        self.datafile_path10 = "data_5kW/data-meth_op4_p_f_p_5.csv"
        self.datafile_path11 = "data_5kW/data-meth_op5_p_f_p_10.csv"
        self.datafile_path12 = "data_5kW/data-meth_op6_p_f_p_15.csv"
        self.datafile_path13 = "data_5kW/data-meth_op7_p_f_p_22.csv"
        self.datafile_path14 = "data_5kW/data-meth_op8_f_p.csv"
        self.datafile_path15 = "data_5kW/data-meth_op9_f_p_f_5.csv"
        self.datafile_path16 = "data_5kW/data-meth_op10_f_p_f_10.csv"
        self.datafile_path17 = "data_5kW/data-meth_op11_f_p_f_15.csv"
        self.datafile_path18 = "data_5kW/data-meth_op12_f_p_f_20.csv"
        # self.datafile_path7 = "data/data-meth_op1_start_p_12kw.csv"
        # self.datafile_path8 = "data/data-meth_op2_start_f_12kw.csv"
        # self.datafile_path9 = "data/data-meth_op3_p_f_12kw.csv"
        # self.datafile_path10 = "data/data-meth_op4_p_f_p_5_12kw.csv"
        # self.datafile_path11 = "data/data-meth_op5_p_f_p_10_12kw.csv"
        # self.datafile_path12 = "data/data-meth_op6_p_f_p_15_12kw.csv"
        # self.datafile_path13 = "data/data-meth_op7_p_f_p_22_12kw.csv"
        # self.datafile_path14 = "data/data-meth_op8_f_p_12kw.csv"
        # self.datafile_path15 = "data/data-meth_op9_f_p_f_5_12kw.csv"
        # self.datafile_path16 = "data/data-meth_op10_f_p_f_10_12kw.csv"
        # self.datafile_path17 = "data/data-meth_op11_f_p_f_15_12kw.csv"
        # self.datafile_path18 = "data/data-meth_op12_f_p_f_20_12kw.csv"
        self.datafile_path19 = "data_5kW/data-meth_validation.csv"
        self.t_cat_standby = 188.2              # °C (catalyst temperature threshold for changing standby data set)
        self.t_cat_startup_cold = 160                   # °C (catalyst temperature threshold for cold start conditions)
        self.t_cat_startup_hot = 350                    # °C (catalyst temperature threshold for hot start conditions)
        self.delta_t_cat_load_change = 10.0     # °C (catalyst temperature margin for changing load data set)
        # time threshold for load change data set, from time = 0
        self.time1_start_p_f = 1201  # simulation step -> 2400 sec
        self.time2_start_f_p = 151  # simulation step -> 300 sec
        self.time_p_f = 210                     # simulation steps for load change (asc) -> 420 sec
        self.time_f_p = 126                     # simulation steps for load change (des) -> 252 sec
        self.time1_p_f_p = 51                   # simulation step -> 100 sec
        self.time2_p_f_p = 151                  # simulation step -> 300 sec
        self.time23_p_f_p = 225                 # simulation step inbetween time2_p_f_p and time3_p_f_p
        self.time3_p_f_p = 301                  # simulation step -> 600 sec
        self.time34_p_f_p = 376                 # simulation step inbetween time3_p_f_p and time4_p_f_p
        self.time4_p_f_p = 451                  # simulation step -> 900 sec
        self.time45_p_f_p = 563                 # simulation step inbetween time4_p_f_p and time5_p_f_p
        self.time5_p_f_p = 675                  # simulation step -> 1348 sec
        self.time1_f_p_f = 51                   # simulation step -> 100 sec
        self.time2_f_p_f = 151                  # simulation step -> 300 sec
        self.time23_f_p_f = 225                 # simulation step inbetween time2_f_p_f and time3_f_p_f
        self.time3_f_p_f = 301                  # simulation step -> 600 sec
        self.time34_f_p_f = 376                 # simulation step inbetween time3_f_p_f and time4_f_p_f
        self.time4_f_p_f = 451                  # simulation step -> 900 sec
        self.time45_f_p_f = 526                 # simulation step inbetween time4_f_p_f and time5_f_p_f
        self.time5_f_p_f = 601                  # simulation step -> 1200 sec
        self.ch4_price = 19.0                   # ct/kWh (incl. CH4 sale and CO2 reduction)
        self.steam_price = 0.7                  # ct/kWh
        self.o2_price = 10                      # ct/Nm³
        self.H_u_CH4 = 35.883                   # MJ/m³ (lower heating value)
        self.H_u_H2 = 10.783                    # MJ/m³ (lower heating value)
        self.h_H2O_evap = 2257                  # kJ/kg (at 1 bar)
        # convert_mol_to_Nm3 = R_uni * T_0 / p_0 = 8.3145J/mol/K * 273.15K / 101325Pa = 0.02241407 Nm3/mol
        self.convert_mol_to_Nm3 = 0.02241407    # For an ideal gas at normal conditions
        self.eta_electrolyzer = 0.65            # electrolyzer efficiency

GLOBAL_PARAMS = GlobalParams()

class ElectricityPrice:
    """
    Class that describes the day-ahead electricity price.
    - Imports Data (from file).
    """

    def __init__(self, current_value: np.ndarray, ahead_1h: np.ndarray) -> None:
        """
        Input:
            current_value: Current electricity price at the day-ahead market in EUR/MWh
            ahead_1h: Electricity price at the day-ahead market 1 hour ahead in EUR/MWh
        """
        self.current_value = current_value
        self.ahead_1h = ahead_1h

class MethState:
    """
    Class that describes the operation states of the methanation plant.
    """

    def __init__(self, standby: int, cooldown: int, startup: int, partial_load: int, full_load: int) -> None:
        """
        :param standby:
        :param cooldown:
        :param startup:
        :param partial_load:
        :param full_load:
        """
        self.standby = standby
        self.cooldown = cooldown
        self.startup = startup
        self.partial_load = partial_load
        self.full_load = full_load

class MethAction:
    """
    Class that describes the actions of the methanation plant agent.
    """

    def __init__(self, standby: int, cooldown: int, startup: int, partial_load: int, full_load: int) -> None:
        """
        :param standby:
        :param cooldown:
        :param startup:
        :param partial_load:
        :param full_load:
        """
        self.standby = standby
        self.cooldown = cooldown
        self.startup = startup
        self.partial_load = partial_load
        self.full_load = full_load


# ----------------------------------------------------------------------------------------------------------------------
print("Get time series values...")
df_price = import_historic_prices(GLOBAL_PARAMS.datafile_path1)
# # Transitions: columns = ["Time [min]", "T_cat [°C]", "n_h2 [mol/s]", "n_ch4 [mol/s]", "m_DE [kg/h]", "Pel [W]"]
df_startup_cold = import_data(GLOBAL_PARAMS.datafile_path2)
df_startup_hot = import_data(GLOBAL_PARAMS.datafile_path3)
df_cooldown = import_data(GLOBAL_PARAMS.datafile_path4)
df_standby_down = import_data(GLOBAL_PARAMS.datafile_path5)
df_standby_up = import_data(GLOBAL_PARAMS.datafile_path6)
df_op1_start_p = import_data(GLOBAL_PARAMS.datafile_path7)
df_op2_start_f = import_data(GLOBAL_PARAMS.datafile_path8)
df_op3_p_f = import_data(GLOBAL_PARAMS.datafile_path9)
df_op4_p_f_p_5 = import_data(GLOBAL_PARAMS.datafile_path10)
df_op5_p_f_p_10 = import_data(GLOBAL_PARAMS.datafile_path11)
df_op6_p_f_p_15 = import_data(GLOBAL_PARAMS.datafile_path12)
df_op7_p_f_p_22 = import_data(GLOBAL_PARAMS.datafile_path13)
df_op8_f_p = import_data(GLOBAL_PARAMS.datafile_path14)
df_op9_f_p_f_5 = import_data(GLOBAL_PARAMS.datafile_path15)
df_op10_f_p_f_10 = import_data(GLOBAL_PARAMS.datafile_path16)
df_op11_f_p_f_15 = import_data(GLOBAL_PARAMS.datafile_path17)
df_op12_f_p_f_20 = import_data(GLOBAL_PARAMS.datafile_path18)
df_validation = import_data(GLOBAL_PARAMS.datafile_path19)

df_price["Time"] = pd.to_datetime(df_price["Time"], format="%d-%m-%Y %H:%M")
time_arr = df_price["Time"].values.astype(float)[:-1]

el_price = df_price["Day-Ahead-price [Euro/MWh]"].values.astype(float) / 10    # Convert Euro/MWh into ct/kWh
el_price_act = el_price[:-1]        # current values
el_price_1h = el_price[1:]        # 1h-ahead values

startup_cold = df_to_nparray(df_startup_cold)
startup_hot = df_to_nparray(df_startup_hot)
cooldown = df_to_nparray(df_cooldown)
standby_down = df_to_nparray(df_standby_down)           # standby dataset for high temperatures to standby
standby_up = df_to_nparray(df_standby_up)               # standby dataset for low temperatures to standby
op1_start_p = df_to_nparray(df_op1_start_p)             # partial load - warming up
op2_start_f = df_to_nparray(df_op2_start_f)           # full load - warming up
op3_p_f = df_to_nparray(df_op3_p_f)                   # Load change: Partial -> Full
op4_p_f_p_5 = df_to_nparray(df_op4_p_f_p_5)         # Load change: Partial -> Full -> Partial (Return after 5 min)
op5_p_f_p_10 = df_to_nparray(df_op5_p_f_p_10)       # Load change: Partial -> Full -> Partial (Return after 10 min)
op6_p_f_p_15 = df_to_nparray(df_op6_p_f_p_15)       # Load change: Partial -> Full -> Partial (Return after 15 min)
op7_p_f_p_22 = df_to_nparray(df_op7_p_f_p_22)       # Load change: Partial -> Full -> Partial (Return after 22 min)
op8_f_p = df_to_nparray(df_op8_f_p)                 # Load change: Full -> Partial
op9_f_p_f_5 = df_to_nparray(df_op9_f_p_f_5)         # Load change: Full -> Partial -> Full (Return after 5 min)
op10_f_p_f_10 = df_to_nparray(df_op10_f_p_f_10)     # Load change: Full -> Partial -> Full (Return after 10 min)
op11_f_p_f_15 = df_to_nparray(df_op11_f_p_f_15)     # Load change: Full -> Partial -> Full (Return after 15 min)
op12_f_p_f_20 = df_to_nparray(df_op12_f_p_f_20)    # Load change: Full -> Partial -> Full (Return after 20 min)
np_validation = df_to_nparray(df_validation)

# TIMESTEPSIZE_OP represents the time step size of the operation data set in seconds
TIME_STEP_SIZE_OP = 2          # in 2 Seconds
# SIM_STEP represents the frequency of taking an action within the operation data time series
SIM_STEP = 1
# TIME_STEP_SIZE_SIM represents the time step size between situations in which the agent can take an agent
# e.g. SIM_STEP = 5, TIME_STEP_SIZE_OP = 2 -> Every TIME_STEP_SIZE_SIM = 10 seconds the agent can take an action
TIME_STEP_SIZE_SIM = SIM_STEP * TIME_STEP_SIZE_OP
# TIME_CONVERSION transposes the TIME_STEP_SIZE_SIM to hours
# For TIME_STEP_SIZE_SIM in seconds: TIME_CONVERSION = 1h/3600s
TIME_CONVERSION = 1 / 3600
# The day-ahead electricity data span the whole time range of data, excluding 1h for 1h-forecast
# TOTAL_SIM_STEPS = int((len(df_price["Time"].values) - 1) / (TIME_STEP_SIZE_SIM * TIME_CONVERSION))
TOTAL_SIM_STEPS = 100015



# # In order to introduce stochasticity into the system
# mu, sigma = 0, 8      # Empirically determined
# s = np.round(np.random.normal(mu, sigma, 1000), 0)
# count, bins, ignored = plt.hist(s, 50, density=True)
#
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
#                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
#          linewidth=2, color='r')
# plt.show()


# ----------------------------------------------------------------------------------------------------------------------
print('Define Environment: 20kW-M')

class MethEnvSimulation:
    """
    Simulation model for 20kW-Methanation
    Memory-based upon experimental time series
    """
    def __init__(self, render_mode="None"):
        super().__init__()

        self.el_price = ElectricityPrice(
            el_price_act,
            el_price_1h,
        )

        # Methanation has 5 states:
        self.M_state = MethState(
            standby=1,
            cooldown=2,
            startup=3,
            partial_load=4,
            full_load=5,
        )

        # Methanation has 5 actions:
        self.M_action = MethAction(
            standby=9,
            cooldown=6,
            startup=2,
            partial_load=10,
            full_load=11,
        )

        self.el_price_act = self.el_price.current_value[0]
        self.el_price_1h = self.el_price.ahead_1h[0]         # electricity price in an hour

        self.Meth_State = self.M_state.cooldown
        # self.Meth_State = self.M_state.partial_load
        self.startup = startup_cold      # current startup data set
        self.standby = standby_down  # current standby data set
        self.partial = op1_start_p  # current partial load data set
        self.full = op2_start_f  # current full load data set

        # self.op = cooldown[-1, :]           # operation point in current data set
        # print(self.op)
        # self.Meth_T_cat = np_validation[0, 1]
        # self.Meth_H2_flow = np_validation[0, 2]
        # self.Meth_CH4_flow = np_validation[0, 3]
        # self.Meth_H2O_flow = np_validation[0, 4]
        # self.Meth_el_heating = np_validation[0, 5]
        # print(self.Meth_T_cat)

        self.Meth_T_cat = np_validation[0, 1]
        self.i = self._get_index(cooldown, self.Meth_T_cat)    # represents index of row in specific operation mode
        self.op = cooldown[self.i, :]                          # operation point in current data set
        self.Meth_H2_flow = self.op[2]
        self.Meth_CH4_flow = self.op[3]
        self.Meth_H2_res_flow = self.op[4]
        self.Meth_H2O_flow = self.op[5]
        self.Meth_el_heating = self.op[6]

        self.ch4_volumeflow = self.Meth_CH4_flow * GLOBAL_PARAMS.convert_mol_to_Nm3
        self.ch4_proceeds = self.ch4_volumeflow * GLOBAL_PARAMS.H_u_CH4 * 1000 * GLOBAL_PARAMS.ch4_price
        self.steam_proceeds = self.Meth_H2O_flow * GLOBAL_PARAMS.h_H2O_evap / 3600 * GLOBAL_PARAMS.steam_price
        self.o2_volumeflow = 1/2 * self.Meth_H2_flow * GLOBAL_PARAMS.convert_mol_to_Nm3 * 3600   # Nm3/s * 3600 s/h
        self.o2_proceeds = self.o2_volumeflow * GLOBAL_PARAMS.o2_price
        self.h2_volumeflow = self.Meth_H2_flow * GLOBAL_PARAMS.convert_mol_to_Nm3
        self.elec_costs = (self.Meth_el_heating / 1000 + self.h2_volumeflow * GLOBAL_PARAMS.H_u_H2
                           * 1000 / GLOBAL_PARAMS.eta_electrolyzer) * self.el_price_act

        self.el_price_records = np.empty(TOTAL_SIM_STEPS)
        self.Meth_State_records = np.empty(TOTAL_SIM_STEPS)
        self.Meth_T_cat_records = np.empty(TOTAL_SIM_STEPS)
        self.Meth_H2_flow_records = np.empty((TOTAL_SIM_STEPS))
        self.Meth_H2_res_flow_records = np.empty((TOTAL_SIM_STEPS))
        self.Meth_CH4_flow_records = np.empty((TOTAL_SIM_STEPS))
        self.Meth_H2O_flow_records = np.empty((TOTAL_SIM_STEPS))
        self.Meth_el_heating_records = np.empty((TOTAL_SIM_STEPS))
        self.Meth_hot_cold_records = np.empty((TOTAL_SIM_STEPS))

        self.k = 0  # counts number of agent steps (every 10 minutes)
        self.j = 0  # counts number steps in specific operation mode (every 10 minutes)
        self.hot_cold = 0   # detects whether startup originates from cold or hot conditions (0=cold, 1=hot)
        self.state_change = False   # True: Action changes state; False: Action does not change state
        self.status_int = self.Meth_State

    def _get_index(self, operation, t_cat):
        """
        :param operation: np.array of the operation mode, in which the timestep occurs
        :param t_cat: catalyst temperature
        :return: idx: index for the starting catalyst temperature
        """
        diff = np.abs(operation[:, 1] - t_cat)
        idx = diff.argmin()
        return idx

    def _perform_sim_step(self, operation, initial_state, next_operation, next_state, step_size, idx, j, change_operation):
        """
        :param operation: np.array of the operation mode, in which the timestep occurs
        :param initial_state: Initial methanation state
        :param next_operation: np.array of the subsequent operation mode (if change_operation == True)
        :param next_state: The final state after reaching total_steps
        :param step_size: step size of one time step
        :param idx: index for the starting catalyst temperature
        :param j: index for the next time step
        :param change_operation: if the subsequent operation differs from the current operation (== True)
        :return: op_range: operation range while timestep, r_state: Methanation state, idx, j
        """
        total_steps = len(operation[:, 1])
        if (idx + j * step_size) < total_steps:
            r_state = initial_state
            op_range = operation[(idx + (j-1) * step_size):(idx + j * step_size), :]
        else:
            r_state = next_state
            time_overhead = (idx + j * step_size) - total_steps
            if time_overhead < step_size:
                # For the time overhead, fill op_range for the timestep with values (next operation/end of the data set)
                op_head = operation[(idx + (j - 1) * step_size):, :]
                if change_operation:
                    idx = time_overhead
                    j = 0
                    op_overhead = next_operation[:idx, :]
                else:
                    op_overhead = np.ones((time_overhead, op_head.shape[1])) * operation[-1, :]
                op_range = np.concatenate((op_head, op_overhead), axis=0)
            else:
                # For the time overhead, fill op_range for the timestep with values at the end of the data set
                op_range = np.ones((step_size, operation.shape[1])) * operation[-1, :]
        return op_range, r_state, idx, j

    def step(self, action):
        k = self.k

        if self.Meth_T_cat <= GLOBAL_PARAMS.t_cat_startup_cold:
            self.hot_cold = 0
        elif self.Meth_T_cat >= GLOBAL_PARAMS.t_cat_startup_hot:
            self.hot_cold = 1

        # print("k =", k, 'action=', action)
        if action == self.M_action.standby:                                                       # action -> Standby
            if self.Meth_State == self.M_state.standby:
                self.j += 1
                # Perform one simulation step
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(self.standby, self.Meth_State,
                                                                                  self.standby, self.Meth_State,
                                                                                  SIM_STEP, self.i, self.j, False)
            else:   # State = cooldown/startup/partial_load/full_load
                # Go to State = Standby
                self.Meth_State = self.M_state.standby
                # Select the standby operation mode
                if self.Meth_T_cat <= GLOBAL_PARAMS.t_cat_standby:
                    self.standby = standby_up
                else:
                    self.standby = standby_down
                # np.random.randint(low=-10, high=10) introduces certain stochasticity in the environment
                self.i = self._get_index(self.standby, self.Meth_T_cat) + int(np.round(np.random.normal(0, 8, 1), 0))
                self.j = 1
                # Perform one simulation step
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(self.standby, self.Meth_State,
                                                                                  self.standby, self.Meth_State,
                                                                                  SIM_STEP, self.i, self.j, False)
        elif action == self.M_action.cooldown:                                                   # action -> Cooldown
            if self.Meth_State == self.M_state.cooldown:
                self.j += 1
                # Perform one simulation step
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(cooldown, self.Meth_State,
                                                                                  cooldown, self.Meth_State,
                                                                                  SIM_STEP, self.i, self.j, False)
            else:   # State = startup/partial_load/full_load
                # Go to State = Cooldown
                self.Meth_State = self.M_state.cooldown
                # Get index of the specific state according to T_cat
                self.i = self._get_index(cooldown, self.Meth_T_cat) + int(np.round(np.random.normal(0, 8, 1), 0))
                self.j = 1
                # Perform one simulation step
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(cooldown, self.Meth_State,
                                                                                  cooldown, self.Meth_State,
                                                                                  SIM_STEP, self.i, self.j, False)
        elif action == self.M_action.startup:                                                     # action -> Startup
            if self.Meth_State == self.M_state.startup:
                self.j += 1
                self.partial = op1_start_p
                # Perform one simulation step
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(self.startup, self.Meth_State,
                                                                                  self.partial, self.M_state.partial_load,
                                                                                  SIM_STEP, self.i, self.j, True)
            elif self.Meth_State == self.M_state.partial_load:
                self.j += 1
                # Perform one simulation step in partial load
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(self.partial, self.Meth_State,
                                                                                  self.partial, self.M_state.partial_load,
                                                                                  SIM_STEP, self.i, self.j, False)
            elif self.Meth_State == self.M_state.full_load:
                self.j += 1
                # Perform one simulation step in full load
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(self.full, self.Meth_State,
                                                                                  self.full, self.M_state.full_load,
                                                                                  SIM_STEP, self.i, self.j, False)
            else:   # State = cooldown/standby
                # Go to State = Startup
                self.Meth_State = self.M_state.startup
                # Select the startup operation mode
                if self.hot_cold == 0:
                    self.startup = startup_cold
                else:   # self.hot_cold ==1
                    self.startup = startup_hot
                self.i = self._get_index(self.startup, self.Meth_T_cat) + int(np.round(np.random.normal(0, 8, 1), 0))
                self.j = 1
                # Perform one simulation step
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(self.startup, self.Meth_State,
                                                                                  self.partial, self.M_state.partial_load,
                                                                                  SIM_STEP, self.i, self.j, True)
        elif action == self.M_action.partial_load:                                           # action -> Partial load
            if self.Meth_State == self.M_state.standby:
                self.j += 1
                # Perform one simulation step in standby
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(self.standby, self.Meth_State,
                                                                                  self.standby, self.Meth_State,
                                                                                  SIM_STEP, self.i, self.j, False)
            elif self.Meth_State == self.M_state.cooldown:
                self.j += 1
                # Perform one simulation step in cooldown
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(cooldown, self.Meth_State,
                                                                                  cooldown, self.Meth_State,
                                                                                  SIM_STEP, self.i, self.j, False)
            elif self.Meth_State == self.M_state.startup:
                self.j += 1
                self.partial = op1_start_p
                # Perform one simulation step in startup
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(self.startup, self.Meth_State,
                                                                                  self.partial, self.M_state.partial_load,
                                                                                  SIM_STEP, self.i, self.j, True)
            elif self.Meth_State == self.M_state.partial_load:
                self.j += 1
                # Perform one simulation step in partial load
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(self.partial, self.Meth_State,
                                                                                  self.partial, self.M_state.partial_load,
                                                                                  SIM_STEP, self.i, self.j, False)
            else:   # State = full_load
                # Go to State = Partial load
                self.Meth_State = self.M_state.partial_load
                # Select the partial_load operation mode
                time_op = self.i + self.j * SIM_STEP          # Simulation step in full_load
                if np.array_equal(self.full, op2_start_f):           # full load operation still warming up or warm up completed
                    if time_op < GLOBAL_PARAMS.time2_start_f_p:
                        self.partial = op1_start_p            # approximation: simple change without temperature changes
                        self.i = self._get_index(self.partial, self.Meth_T_cat)
                        self.j = 1
                    else:
                        self.partial = op8_f_p
                        self.i = 0
                        self.j = 1
                elif np.array_equal(self.full, op3_p_f):             # full load operation in load change partial -> full
                    if time_op < GLOBAL_PARAMS.time1_p_f_p:          # approximation: simple go back to original state
                        self.partial = op8_f_p
                        self.i = 12000                                   # fully developed operation
                        self.j = 100
                        self.Meth_T_cat = op8_f_p[-1, 1]
                    elif GLOBAL_PARAMS.time1_p_f_p < time_op < GLOBAL_PARAMS.time2_p_f_p:
                        self.partial = op4_p_f_p_5
                        self.j += 1
                    elif GLOBAL_PARAMS.time2_p_f_p < time_op < GLOBAL_PARAMS.time_p_f:
                        self.partial = op4_p_f_p_5
                        self.i = GLOBAL_PARAMS.time2_p_f_p
                        self.j = 0
                    elif GLOBAL_PARAMS.time_p_f < time_op < GLOBAL_PARAMS.time34_p_f_p:
                        self.partial = op5_p_f_p_10
                        self.i = GLOBAL_PARAMS.time3_p_f_p
                        self.j = 0
                    elif GLOBAL_PARAMS.time34_p_f_p < time_op < GLOBAL_PARAMS.time45_p_f_p:
                        self.partial = op6_p_f_p_15
                        self.i = GLOBAL_PARAMS.time4_p_f_p
                        self.j = 0
                    elif GLOBAL_PARAMS.time45_p_f_p < time_op < GLOBAL_PARAMS.time5_p_f_p:
                        self.partial = op7_p_f_p_22
                        self.i = GLOBAL_PARAMS.time5_p_f_p
                        self.j = 0
                    else: # time_op > GLOBAL_PARAMS.time5_p_f_p
                        self.partial = op8_f_p
                        self.i = 0
                        self.j = 1
                elif np.array_equal(self.full, op9_f_p_f_5):       # full load operation in load change full -> partial -> full after 5m
                        self.partial = op8_f_p
                        self.i = 0
                        self.j = 1
                elif np.array_equal(self.full, op10_f_p_f_10):       # full load operation in load change full -> partial -> full after 10m
                        self.partial = op8_f_p
                        self.i = 0
                        self.j = 1
                elif np.array_equal(self.full, op11_f_p_f_15):       # full load operation in load change full -> partial -> full after 15m
                        self.partial = op8_f_p
                        self.i = 0
                        self.j = 1
                else:  # self.full == op12_f_p_f_22    full load operation in load change full -> partial -> full after 22m
                    self.partial = op8_f_p
                    self.i = 0
                    self.j = 1

                # Perform one simulation step
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(self.partial, self.Meth_State,
                                                                                  self.partial, self.M_state.partial_load,
                                                                                  SIM_STEP, self.i, self.j, False)
        elif action == self.M_action.full_load:                                                 # action -> Full load
            if self.Meth_State == self.M_state.standby:
                self.j += 1
                # Perform one simulation step in standby
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(self.standby, self.Meth_State,
                                                                                  self.standby, self.Meth_State,
                                                                                  SIM_STEP, self.i, self.j, False)
            elif self.Meth_State == self.M_state.cooldown:
                self.j += 1
                # Perform one simulation step in cooldown
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(cooldown, self.Meth_State,
                                                                                  cooldown, self.Meth_State,
                                                                                  SIM_STEP, self.i, self.j, False)
            elif self.Meth_State == self.M_state.startup:
                self.j += 1
                self.partial = op1_start_p
                # Perform one simulation step in startup
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(self.startup, self.Meth_State,
                                                                                  self.partial, self.M_state.partial_load,
                                                                                  SIM_STEP, self.i, self.j, True)
            elif self.Meth_State == self.M_state.full_load:
                self.j += 1
                # Perform one simulation step in partial load
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(self.full, self.Meth_State,
                                                                                  self.full, self.M_state.full_load,
                                                                                  SIM_STEP, self.i, self.j, False)
            else:  # State = partial_load
                # Go to State = Full load
                self.Meth_State = self.M_state.full_load
                # Select the full_load operation mode
                time_op = self.i + self.j * SIM_STEP  # Simulation step in partial_load
                if np.array_equal(self.partial, op1_start_p):  # partial load operation still warming up or warm up completed
                    if time_op < GLOBAL_PARAMS.time1_start_p_f:
                        self.full = op2_start_f  # approximation: simple change without temperature changes
                        self.i = 0
                        self.j = 1
                    else:
                        self.full = op3_p_f
                        self.i = 0
                        self.j = 1
                elif np.array_equal(self.partial, op8_f_p):  # full load operation in load change full -> partial
                    if time_op < GLOBAL_PARAMS.time1_f_p_f:  # approximation: simple go back to original state
                        self.full = op3_p_f
                        self.i = 12000  # fully developed operation
                        self.j = 100
                        self.Meth_T_cat = op3_p_f[-1, 1]

                    elif GLOBAL_PARAMS.time1_f_p_f < time_op < GLOBAL_PARAMS.time_f_p:
                        self.full = op9_f_p_f_5
                        self.j += 1
                    elif GLOBAL_PARAMS.time_f_p < time_op < GLOBAL_PARAMS.time23_f_p_f:
                        self.full = op9_f_p_f_5
                        self.i = GLOBAL_PARAMS.time2_f_p_f
                        self.j = 0
                    elif GLOBAL_PARAMS.time23_f_p_f < time_op < GLOBAL_PARAMS.time34_f_p_f:
                        self.full = op10_f_p_f_10
                        self.i = GLOBAL_PARAMS.time3_f_p_f
                        self.j = 0
                    elif GLOBAL_PARAMS.time34_f_p_f < time_op < GLOBAL_PARAMS.time45_f_p_f:
                        self.full = op11_f_p_f_15
                        self.i = GLOBAL_PARAMS.time4_f_p_f
                        self.j = 0
                    elif GLOBAL_PARAMS.time45_f_p_f < time_op < GLOBAL_PARAMS.time5_f_p_f:
                        self.full = op12_f_p_f_20
                        self.i = GLOBAL_PARAMS.time5_f_p_f
                        self.j = 0
                    else:  # time_op > GLOBAL_PARAMS.time5_f_p_f
                        self.full = op3_p_f
                        self.i = 0
                        self.j = 1
                elif np.array_equal(self.partial, op4_p_f_p_5):  # partial load operation in load change partial -> full -> partial after 5m
                    self.full = op3_p_f
                    self.i = 0
                    self.j = 1
                elif np.array_equal(self.partial, op5_p_f_p_10):  # partial load operation in load change partial -> full -> partial after 10m
                    self.full = op3_p_f
                    self.i = 0
                    self.j = 1
                elif np.array_equal(self.partial, op6_p_f_p_15):  # partial load operation in load change partial -> full -> partial after 15m
                    self.full = op3_p_f
                    self.i = 0
                    self.j = 1
                else:  # self.full == op7_f_p_f_22    partial load operation in load change partial -> full -> partial after 22m
                    self.partial = op8_f_p
                    self.i = 0
                    self.j = 1

                # Perform one simulation step
                self.op, self.Meth_State, self.i, self.j = self._perform_sim_step(self.full, self.Meth_State,
                                                                                  self.full, self.M_state.full_load,
                                                                                  SIM_STEP, self.i, self.j, False)

        # self.el_price_act = self.el_price.current_value[math.floor(k * TIME_STEP_SIZE_SIM * TIME_CONVERSION)]
        # self.el_price_1h = self.el_price.ahead_1h[math.floor(k * TIME_STEP_SIZE_SIM * TIME_CONVERSION)]  # electricity price in an hour
        self.Meth_T_cat = self.op[-1, 1]  # Last value in self.op = new catalyst temperature

        # Form the averaged values of species flow and electrical heating during time step
        self.Meth_H2_flow = np.average(self.op[:, 2])
        self.Meth_CH4_flow = np.average(self.op[:, 3])
        self.Meth_H2_res_flow = np.average(self.op[:, 4])
        self.Meth_H2O_flow = np.average(self.op[:, 5])
        self.Meth_el_heating = np.average(self.op[:, 6])

        self.el_price_records[k] = self.el_price_act
        self.Meth_State_records[k] = self.Meth_State
        self.Meth_T_cat_records[k] = self.Meth_T_cat
        self.Meth_H2_flow_records[k] = self.Meth_H2_flow
        self.Meth_H2_res_flow_records[k] = self.Meth_H2_res_flow
        self.Meth_CH4_flow_records[k] = self.Meth_CH4_flow
        self.Meth_H2O_flow_records[k] = self.Meth_H2O_flow
        self.Meth_el_heating_records[k] = self.Meth_el_heating
        self.Meth_hot_cold_records[k] = self.hot_cold

        self.k += 1

class PTGEnv:
    """Custom Environment implementing the Gymnasium interface for PtG dispatch optimization."""

    metadata = {"render_modes": ["None"]}

    def __init__(self, dict_input, train_or_eval = "train", render_mode="None"):
        """
            Initialize the PtG environment for training or evaluation
            :param dict_input: Dictionary containing energy market data, process data, and training configurations
            :param train_or_eval: Specifies if detailed state descriptions are provided for evaluation ("eval") or not ("train", default for training)
            :param render_mode: Specifies the rendering mode
        """
        super().__init__()

        global ep_index

        # Unpack data from dictionary
        self.__dict__.update(dict_input)

        assert train_or_eval in ["train", "eval"], f'ptg_gym_env.py error: train_or_eval must be either "train" or "eval".'
        self.train_or_eval = train_or_eval

        # Methanation plant process states: [0, 1, 2, 3, 4]
        self.M_state = {
            'standby': self.ptg_standby,
            'cooldown': self.ptg_cooldown,
            'startup': self.ptg_startup,
            'partial_load': self.ptg_partial_load,
            'full_load': self.ptg_full_load,
        }

        # Initialize dynamic simulation variables and time tracking
        if isinstance(self.eps_ind, np.ndarray):            # Training environment scenario
            self.act_ep_h = int(self.eps_ind[ep_index] * self.eps_len_d * 24)
            self.act_ep_d = int(self.eps_ind[ep_index] * self.eps_len_d)
            ep_index += 1                                   # Select next data subset for subsequent episode
        else:                                               # Validation or test environments
            self.act_ep_h, self.act_ep_d = 0, 0
        self.time_step_size_sim = self.sim_step
        self.step_size = int(self.time_step_size_sim / self.time_step_op)
        self.clock_hours = 0 * self.time_step_size_sim / 3600   # in [h]
        self.clock_days = self.clock_hours / 24                 # in [d]
        # self.act_ep_h = self.act_ep_h + self.price_past         # Add the number of past values to the current index
        # self.act_ep_d = self.act_ep_d + 1                       # Add 1 to the current index to account for the current value   

        self._initialize_datasets()
        self._initialize_op_rew()
        self._initialize_action_space()
        self._initialize_observation_space()
        self._normalize_observations()

        if self.scenario == 3: self.b_s3 = 1
        else: self.b_s3 = 0

        self.render_mode = render_mode
     
       
    def _initialize_datasets(self):
        """Initialize data sets and temporal encoding"""
        # self.e_r_b: np.array that stores elec. price data, potential reward, and boolean identifier
        #       Dimensions = [Type of data] x [No. of values] x [historical values]
        #           Type of data = [el_price, pot_rew, part_full_b]
        #           No. of values = price_ahead + price_past
        #           historical values = No. of values in the electricity price data set
        # e.g. e_r_b_train[0, 12, 156] represents the current value of the electricity price [0,-,-] at the
        # 156ths entry of the electricity price data set 
        self.e_r_b_act = self.e_r_b[:, :, self.act_ep_h]   # values [-12h, ..., 0h, ..., 12h)]

        # self.g_e: np.array that stores gas and EUA price data
        #       Dimensions = [Type of data] x [No. of day-ahead values] x [historical values]
        #           Type of data = [gas_price, eua_price]
        #           No. of day-ahead values = 2 (today and tomorrow)
        #           historical values = No. of values in the price data set
        self.g_e_act = self.g_e[:, :, self.act_ep_d]        # values [0h, 24h]

        # Initialize gas and EUA price arrays
        self.num_gas_eua = 3                                                # Number of gas and EUA prices (before 12 hours, current, and in 12 hours)
        self.gas_eua_price_d = np.zeros((2, self.num_gas_eua))              # Gas/EUA prices [-12h, 0, 12h]
        self.gas_eua_price_d[:, 0] = self.g_e[:, 0, self.act_ep_d-1]        # Since the data sets start at 0:00, the first entry is the value of the previous day 
        self.gas_eua_price_d[:, 1] = self.g_e[:, 0, self.act_ep_d]
        self.gas_eua_price_d[:, 2] = self.g_e[:, 0, self.act_ep_d]

        # # Temporal encoding for time step within an hour (sine-cosine transformation)
        # self.temp_h_enc_sin = math.sin(2 * math.pi * self.clock_hours)
        # self.temp_h_enc_cos = math.cos(2 * math.pi * self.clock_hours)

    def _initialize_op_rew(self):
        """Initialize methanation operation and reward constituents"""
        # Methanation operation
        self.Meth_State = self.M_state['cooldown']
        self.Meth_states = list(self.M_state.keys())                # Methanation state space
        self.current_state = 'cooldown'                             # Current state as string
        self.standby = self.standby_down                            # Current standby data set
        self.standby_op = 'standby_down'
        self.startup = self.startup_cold                            # Current startup data set
        self.startup_op = 'startup_cold'
        self.partial = self.op1_start_p                             # Current partial load data set
        self.part_op = 'op1_start_p'                                # Track partial load conditions
        self.partial_map = {
            'op1_start_p': self.op1_start_p,
            'op8_f_p': self.op8_f_p,
            'op4_p_f_p_5': self.op4_p_f_p_5,
            'op5_p_f_p_10': self.op5_p_f_p_10,
            'op6_p_f_p_15': self.op6_p_f_p_15,
            'op7_p_f_p_22': self.op7_p_f_p_22,
        }
        self.full = self.op2_start_f                                # Current full load data set
        self.full_op = 'op2_start_f'                                # Track full load conditions
        self.full_map = {
            'op2_start_f': self.op2_start_f,
            'op3_p_f': self.op3_p_f,
            'op9_f_p_f_5': self.op9_f_p_f_5,
            'op10_f_p_f_10': self.op10_f_p_f_10,
            'op11_f_p_f_15': self.op11_f_p_f_15,
            'op12_f_p_f_20': self.op12_f_p_f_20,
        }
        self.Meth_T_cat = 16                                        # Initial catalyst temperature [°C] 
        self.i = self._get_index(self.cooldown, self.Meth_T_cat)    # Index for operation
        self.j = 0                                                  # Step counter for operation
        self.op = self.cooldown[self.i, :]                          # Current operation point               
        keys = ['H2_flow', 'CH4_flow', 'H2_res_flow', 'H2O_flow', 'el_heating'] 
        for i, key in enumerate(keys, start=2): setattr(self, f'Meth_{key}', self.op[i])
        keys = ['T_cat', 'H2_flow', 'CH4_flow', 'H2_res_flow', 'H2O_flow', 'el_heating']   
        self.op_seq = np.ones((self.seq_length, len(keys)+1)) * self.op               # Broadcasts vector across rows                         
        for i, key in enumerate(keys, start=1): setattr(self, f'Meth_{key}_seq', self.op_seq[:, i])
        self.hot_cold = 0                   # Detect startup conditions (0=cold, 1=hot)
        self.state_change = False           # Track changes in methanation state Meth_State
        self.r_0 = self.reward_level[0]     # Reward level

        # Reward constituents
        self.ch4_volumeflow, self.h2_res_volumeflow, self.Q_ch4, self.Q_h2_res, self.ch4_revenues = (0.0,) * 5
        self.power_chp, self.chp_revenues, self.Q_steam, self.steam_revenues, self.h2_volumeflow = (0.0,) * 5
        self.o2_volumeflow, self.o2_revenues, self.Meth_CO2_mass_flow, self.eua_revenues = (0.0,) * 4
        self.elec_costs_heating, self.load_elec, self.elec_costs_electrolyzer, self.elec_costs = (0.0,) * 4
        self.water_elec, self.water_costs, self.rew, self.cum_rew = (0.0,) * 4
        self.eta_electrolyzer = 0.02        # Initial electrolyzer efficiency
        self.cum_rew = 0                    # Cumulative reward

        # Info object and step counter
        self.info = {}                      # Info for evaluation
        self.k = 0                          # Step counter
        
    def _initialize_action_space(self):
        """Initialize the action space for plant operations""" 
        self.actions = ['standby', 'cooldown', 'startup', 'partial_load', 'full_load']
        self.current_action = 'cooldown'                    # Aligned with the real-world plant
        if self.action_type == "discrete":
            self.action_space = gym.spaces.Discrete(5)
        elif self.action_type == "continuous":
            self.act_b = [-1, 1]                            # Lower and upper bounds of the value range [low, up]
            # For discretization of continuous actions:
            # -> if self.prob_thre[i-1] < action < self.prob_thre[i]: -> Pick self.actions[i]
            # self.prob_ival: Distance for discrete probability intervals for taken specific action
            self.prob_ival = (self.act_b[1] - self.act_b[0]) / len(self.actions) 
            # self.prob_thre: Number of thresholds for the intervals: [l_b, l_b + ival,..., u_b]      
            self.prob_thre = np.ones((len(self.actions) + 1,))  
            for ival in range(len(self.prob_thre)):
                self.prob_thre[ival] = self.act_b[0] + ival * self.prob_ival
            self.action_space = gym.spaces.Box(low=self.act_b[0], high=self.act_b[1], shape=(1,), dtype=np.float32)
        else:
            assert False, f"ptg_gym_env.py error: invalid action type ({self.action_type}) - must match ['discrete', 'continuous']!"

    def _initialize_observation_space(self):
        """Define observation space"""
        b_norm = [0, 1]     # Normalized lower and upper bounds [low, up]
        
        self.observation_space = spaces.Dict(
            {
                "Elec_Price": spaces.Box(low=b_norm[0] * np.ones((self.price_ahead + self.price_past,)),
                                        high=b_norm[1] * np.ones((self.price_ahead + self.price_past,)), dtype=np.float64),
                "Gas_Price": spaces.Box(low=b_norm[0] * np.ones((self.num_gas_eua,)),
                                        high=b_norm[1] * np.ones((self.num_gas_eua,)), dtype=np.float64),
                "EUA_Price": spaces.Box(low=b_norm[0] * np.ones((self.num_gas_eua,)),
                                        high=b_norm[1] * np.ones((self.num_gas_eua,)), dtype=np.float64),
                "T_CAT": spaces.Box(low=b_norm[0], high=b_norm[1], shape=(self.seq_length,), dtype=np.float64),
                "H2_in_MolarFlow": spaces.Box(low=b_norm[0], high=b_norm[1], shape=(self.seq_length,), dtype=np.float64),
                "CH4_syn_MolarFlow": spaces.Box(low=b_norm[0], high=b_norm[1], shape=(self.seq_length,), dtype=np.float64),
                "H2_res_MolarFlow": spaces.Box(low=b_norm[0], high=b_norm[1], shape=(self.seq_length,), dtype=np.float64),
                "H2O_DE_MassFlow": spaces.Box(low=b_norm[0], high=b_norm[1], shape=(self.seq_length,), dtype=np.float64),
                "Elec_Heating": spaces.Box(low=b_norm[0], high=b_norm[1], shape=(self.seq_length,), dtype=np.float64),
            }
        )
        
    def _normalize_observations(self):
        """Normalize observations using standardization"""
        
        self.el_n = (self.e_r_b_act[0, :] - self.el_l_b) / (self.el_u_b - self.el_l_b)
        self.gas_n = (self.gas_eua_price_d[0, :] - self.gas_l_b) / (self.gas_u_b - self.gas_l_b)
        self.eua_n = (self.gas_eua_price_d[1, :] - self.eua_l_b) / (self.eua_u_b - self.eua_l_b)
        self.Meth_T_cat_n = (self.Meth_T_cat_seq - self.T_l_b) / (self.T_u_b - self.T_l_b)
        self.Meth_H2_flow_n = (self.Meth_H2_flow_seq - self.h2_l_b) / (self.h2_u_b - self.h2_l_b)
        self.Meth_CH4_flow_n = (self.Meth_CH4_flow_seq - self.ch4_l_b) / (self.ch4_u_b - self.ch4_l_b)
        self.Meth_H2_res_flow_n = (self.Meth_H2_res_flow_seq - self.h2_res_l_b) / (self.h2_res_u_b - self.h2_res_l_b)
        self.Meth_H2O_flow_n = (self.Meth_H2O_flow_seq - self.h2o_l_b) / (self.h2o_u_b - self.h2o_l_b)
        self.Meth_el_heating_n = (self.Meth_el_heating_seq - self.heat_l_b) / (self.heat_u_b - self.heat_l_b)

    def get_obs(self):
        """Retrieve the current observations from the environment"""
        return {
            "Elec_Price": np.array(self.el_n, dtype=np.float64),
            "Gas_Price": np.array(self.gas_n, dtype=np.float64),
            "EUA_Price": np.array(self.eua_n, dtype=np.float64),
            "T_CAT": np.array(self.Meth_T_cat_n, dtype=np.float64),
            "H2_in_MolarFlow": np.array(self.Meth_H2_flow_n, dtype=np.float64),
            "CH4_syn_MolarFlow": np.array(self.Meth_CH4_flow_n, dtype=np.float64),
            "H2_res_MolarFlow": np.array(self.Meth_H2_res_flow_n, dtype=np.float64),
            "H2O_DE_MassFlow": np.array(self.Meth_H2O_flow_n, dtype=np.float64),
            "Elec_Heating": np.array(self.Meth_el_heating_n, dtype=np.float64),
        }
    
    def _get_info(self, state_c):
        """Retrieve additional details or metadata about the environment"""
        return {
            "step": self.k,
            "el_price_act": self.e_r_b_act[0, 12],
            "gas_price_act": self.g_e_act[0, 0],
            "eua_price_act": self.g_e_act[1, 0],
            "Meth_State": self.Meth_State,
            "Meth_Action": self.current_action,
            "Meth_Hot_Cold": self.hot_cold,
            "Meth_T_cat": self.Meth_T_cat,
            "Meth_H2_flow": self.Meth_H2_flow,
            "Meth_CH4_flow": self.Meth_CH4_flow,
            "Meth_H2O_flow": self.Meth_H2O_flow,
            "Meth_el_heating": self.Meth_el_heating,
            "ch4_revenues [ct/h]": self.ch4_revenues,
            "steam_revenues [ct/h]": self.steam_revenues,
            "o2_revenues [ct/h]": self.o2_revenues,
            "eua_revenues [ct/h]": self.eua_revenues,
            "chp_revenues [ct/h]": self.chp_revenues,
            "elec_costs_heating [ct/h]": -self.elec_costs_heating,
            "elec_costs_electrolyzer [ct/h]": -self.elec_costs_electrolyzer,
            "water_costs [ct/h]": -self.water_costs,
            "reward [ct]": self.rew,
            "cum_reward": self.cum_rew,
            "Pot_Reward": self.e_r_b_act[1, 12],
            "Part_Full": self.e_r_b_act[2, 12],
            "state_c": state_c,
        }

    def _get_reward(self):
        """Calculate the reward based on the current revenues and costs"""

        # Gas revenues (Scenario 1+2):          If Scenario == 3: self.gas_price_h[0] = 0
        self.ch4_volumeflow = self.Meth_CH4_flow * self.convert_mol_to_Nm3              # in [Nm³/s]
        self.h2_res_volumeflow = self.Meth_H2_res_flow * self.convert_mol_to_Nm3        # in [Nm³/s]
        self.Q_ch4 = self.ch4_volumeflow * self.H_u_CH4 * 1000                          # Thermal power of methane in [kW]
        self.Q_h2_res = self.h2_res_volumeflow * self.H_u_H2 * 1000                     # Thermal power of residual hydrogen in [kW]
        self.ch4_revenues = (self.Q_ch4 + self.Q_h2_res) * self.g_e_act[0, 0]           # SNG revenues in [ct/h]

        # CHP revenues (Scenario 3):               If Scenario == 3: self.b_s3 = 1 else self.b_s3 = 0
        self.power_chp = self.Q_ch4 * self.eta_CHP * self.b_s3                          # Electrical power of the CHP in [kW]
        self.Q_chp = self.Q_ch4 * (1 - self.eta_CHP) * self.b_s3                        # Thermal power of the produced steam in the CHP in [kW]
        self.chp_revenues = self.power_chp * self.eeg_el_price                          # EEG tender revenues in [ct/h]

        # Steam revenues (Scenario 1+2+3):          If Scenario != 3: self.Q_chp = 0
        self.Q_steam = self.Meth_H2O_flow * (self.dt_water * self.cp_water + self.h_H2O_evap) / 3600    # Thermal power of the produced steam in the methanation plant in [kW]
        self.steam_revenues = (self.Q_steam + self.Q_chp) * self.heat_price                             # in [ct/h]

        # Oxygen revenues (Scenario 1+2+3):
        self.h2_volumeflow = self.Meth_H2_flow * self.convert_mol_to_Nm3                # in [Nm³/s]
        self.o2_volumeflow = 1 / 2 * self.h2_volumeflow * 3600                          # in [Nm³/h] = [Nm³/s * 3600 s/h]
        self.o2_revenues = self.o2_volumeflow * self.o2_price                           # Oxygen revenues in [ct/h]

        # EUA revenues (Scenario 1+2):              If Scenario == 3: self.eua_price_h[0] = 0
        self.Meth_CO2_mass_flow = self.Meth_CH4_flow * self.Molar_mass_CO2 / 1000                       # Consumed CO2 mass flow in [kg/s]
        self.eua_revenues = self.Meth_CO2_mass_flow / 1000 * 3600 * self.g_e_act[1, 0] * 100            # EUA revenues in ct/h = kg/s * t/1000kg * 3600 s/h * €/t * 100 ct/€

        # Linear regression model for LHV efficiency of a 6 MW electrolyzer
        # Costs for electricity:
        self.elec_costs_heating = self.Meth_el_heating / 1000 * self.e_r_b_act[0, 12]    # Electricity costs for methanation heating in [ct/h]
        self.load_elec = self.h2_volumeflow / self.max_h2_volumeflow                    # Electrolyzer load
        if self.load_elec < self.min_load_electrolyzer:
            self.eta_electrolyzer = 0.02
        else:
            self.eta_electrolyzer = (0.598 - 0.325 * self.load_elec ** 2 + 0.218 * self.load_elec ** 3 +
                                     0.01 * self.load_elec ** (-1) - 1.68 * 10 ** (-3) * self.load_elec ** (-2) +
                                     2.51 * 10 ** (-5) * self.load_elec ** (-3))
        self.elec_costs_electrolyzer = self.h2_volumeflow * self.H_u_H2 * 1000 / self.eta_electrolyzer * \
                                       self.e_r_b_act[0, 12]                             # Electricity costs for water electrolysis in [ct/h]
        self.elec_costs = self.elec_costs_heating + self.elec_costs_electrolyzer

        # Costs for water consumption:
        self.water_elec = self.Meth_H2_flow * self.Molar_mass_H2O / 1000 * 3600                         # Water demand of the electrolyzer in [kg/h] (1 mol water is consumed for producing 1 mol H2)
        self.water_costs = (self.Meth_H2O_flow + self.water_elec) / self.rho_water * self.water_price   # Water costs in [ct/h] = [kg/h / (kg/m³) * ct/m³]

        # Reward:
        self.rew = (self.ch4_revenues + self.chp_revenues + self.steam_revenues + self.eua_revenues +
                    self.o2_revenues - self.elec_costs - self.water_costs) * self.time_step_size_sim / 3600

        self.cum_rew += self.rew

        if self.state_change == True: self.rew -= self.r_0 * self.state_change_penalty

        rew_norm = (self.rew - self.rew_l_b) / (self.rew_u_b - self.rew_l_b)  # Normalize reward

        return rew_norm

    def step(self, act_state_c):
        action = act_state_c[0]  # Action is passed as a parameter to avoid deepcopy of the environment in MCTS
        state_c = act_state_c[1]  # State is passed as a parameter to avoid deepcopy of the environment in MCTS
        # Unpacking from state_c dict
        for key in ['i', 'j', 'k', 'Meth_State', 'Meth_T_cat', 'standby_op', 'startup_op', 'part_op', 'full_op']:
            setattr(self, key, state_c[key])

        # Standby and startup mappings
        self.standby = self.standby_down if self.standby_op == 'standby_down' else self.standby_up
        self.startup = self.startup_cold if self.startup_op == 'startup_cold' else self.startup_hot

        # Partial operation
        try:
            self.partial = self.partial_map[self.part_op]
        except KeyError:
            raise ValueError(f"Partial operation '{self.part_op}' not recognized.")

        # Full operation
        try:
            self.full = self.full_map[self.full_op]
        except KeyError:
            raise ValueError(f"Full operation '{self.full_op}' not recognized.")

        k = self.k

        if self.Meth_T_cat <= self.t_cat_startup_cold:
            self.hot_cold = 0
        elif self.Meth_T_cat >= self.t_cat_startup_hot:
            self.hot_cold = 1

        # previous_state = self.Meth_State

        if self.action_type == "discrete":
            self.current_action = self.actions[action]
        elif self.action_type == "continuous":
            # For discretization of continuous actions:
            # -> if self.prob_thre[i-1] < action < self.prob_thre[i]: -> Pick self.actions[i]
            check_ival = self.prob_thre > action
            for ival in range(len(check_ival)):
                if check_ival[ival]:
                    self.current_action = self.actions[int(ival - 1)]
                    break
        else:
            assert False, f"ptg_gym_env.py error: invalid action type ({self.action_type}) - ['discrete', 'continuous']!"

        self.current_state = self.Meth_states[self.Meth_State]

        # When the agent takes an action, the environment's reaction depends on the current methanation state.
        # NOTE:
        # ptg_gym_env uses match-case conditions to determine the environment's response. This structure is similar to if-else conditions but offers better clarity.
        # Although match-case might slightly reduce performance compared to other selection methods (e.g., nested dictionaries), preliminary performance tests show 
        # it performs comparably for up to 100 million time steps. 
        # The primary performance bottleneck is memory access and data transfer when loading energy market and PtG process data, 
        # even with memory optimization and caching in place.
        match self.current_action:
            case 'standby':
                match self.current_state:
                    case 'standby':
                        self.op, self.Meth_State, self.i, self.j = self._cont(self.standby, self.Meth_State,
                                                                              self.standby, self.Meth_State, False)
                    case _:
                        self.op, self.Meth_State, self.i, self.j = self._standby()
            case 'cooldown':
                match self.current_state:
                    case 'cooldown':
                        self.op, self.Meth_State, self.i, self.j = self._cont(self.cooldown, self.Meth_State,
                                                                              self.cooldown, self.Meth_State, False)
                    case _:
                        self.op, self.Meth_State, self.i, self.j = self._cooldown()
            case 'startup':
                match self.current_state:
                    case 'startup':
                        self.op, self.Meth_State, self.i, self.j = self._cont(self.startup, self.Meth_State,
                                                                              self.partial,
                                                                              self.M_state['partial_load'], True)
                    case 'partial_load':
                        self.op, self.Meth_State, self.i, self.j = self._cont(self.partial, self.Meth_State,
                                                                              self.partial,
                                                                              self.M_state['partial_load'],
                                                                              False)
                    case 'full_load':
                        self.op, self.Meth_State, self.i, self.j = self._cont(self.full, self.Meth_State,
                                                                              self.full, self.M_state['full_load'],
                                                                              False)
                    case _:
                        self.op, self.Meth_State, self.i, self.j = self._startup()
            case 'partial_load':
                match self.current_state:
                    case 'standby':
                        self.op, self.Meth_State, self.i, self.j = self._cont(self.standby, self.Meth_State,
                                                                              self.standby, self.Meth_State, False)
                    case 'cooldown':
                        self.op, self.Meth_State, self.i, self.j = self._cont(self.cooldown, self.Meth_State,
                                                                              self.cooldown, self.Meth_State, False)
                    case 'startup':
                        self.op, self.Meth_State, self.i, self.j = self._cont(self.startup, self.Meth_State,
                                                                              self.partial,
                                                                              self.M_state['partial_load'],
                                                                              True)
                    case 'partial_load':
                        self.op, self.Meth_State, self.i, self.j = self._cont(self.partial, self.Meth_State,
                                                                              self.partial,
                                                                              self.M_state['partial_load'],
                                                                              False)
                    case _:
                        self.op, self.Meth_State, self.i, self.j = self._partial()
            case 'full_load':
                match self.current_state:
                    case 'standby':
                        self.op, self.Meth_State, self.i, self.j = self._cont(self.standby, self.Meth_State,
                                                                              self.standby, self.Meth_State, False)
                    case 'cooldown':
                        self.op, self.Meth_State, self.i, self.j = self._cont(self.cooldown, self.Meth_State,
                                                                              self.cooldown, self.Meth_State, False)
                    case 'startup':
                        self.op, self.Meth_State, self.i, self.j = self._cont(self.startup, self.Meth_State,
                                                                              self.partial,
                                                                              self.M_state['partial_load'],
                                                                              True)
                    case 'full_load':
                        self.op, self.Meth_State, self.i, self.j = self._cont(self.full, self.Meth_State,
                                                                              self.full, self.M_state['full_load'],
                                                                              False)
                    case _:  # Partial Load
                        self.op, self.Meth_State, self.i, self.j = self._full()
            case _:
                assert False, f"ptg_gym_env.py error: invalid action ({self.current_action}) - ['standby', 'cooldown', 'startup', 'partial_load', 'full_load']!"

        self.clock_hours = (k + 1) * self.time_step_size_sim / 3600
        self.clock_days = self.clock_hours / 24
        h_step = math.floor(self.clock_hours)
        d_step = math.floor(self.clock_days)
        self.e_r_b_act = self.e_r_b[:, :, self.act_ep_h + h_step]
        self.g_e_act = self.g_e[:, :, self.act_ep_d + d_step]

        self.gas_eua_price_d[:, 1] = self.g_e_act[:, 0]        # Current gas/EUA price

        if self.clock_days % 0.5 == 0 and self.clock_days % 1 != 0:
            self.gas_eua_price_d[:, 0] = self.gas_eua_price_d[:, 1]         # At noon, the value before 12h becomes the current gas/EUA price
            self.gas_eua_price_d[:, 2] = self.g_e_act[:, 1]    # At noon, the value in 12h becomes the next-day gas/EUA price
        
        # self.temp_h_enc_sin = math.sin(2 * math.pi * self.clock_hours)
        # self.temp_h_enc_cos = math.cos(2 * math.pi * self.clock_hours)

        self.Meth_T_cat = self.op[-1, 1]    # Last value in self.op equals the new catalyst temperature
        # Average the species flow and electric heating values over the simulation time step
        self.Meth_H2_flow = np.average(self.op[:, 2])
        self.Meth_CH4_flow = np.average(self.op[:, 3])
        self.Meth_H2_res_flow = np.average(self.op[:, 4])
        self.Meth_H2O_flow = np.average(self.op[:, 5])
        self.Meth_el_heating = np.average(self.op[:, 6])

        # Store past process data sequence
        # For each column, sample backwards from the last entry
        self.op_seq = np.array([self.op[::-1, col][::self.seq_step][:self.seq_length] for col in range(self.op.shape[1])]).T
        self.Meth_T_cat_seq = self.op_seq[:, 1]
        self.Meth_H2_flow_seq = self.op_seq[:, 2]
        self.Meth_CH4_flow_seq = self.op_seq[:, 3]
        self.Meth_H2_res_flow_seq = self.op_seq[:, 4]
        self.Meth_H2O_flow_seq = self.op_seq[:, 5]
        self.Meth_el_heating_seq = self.op_seq[:, 6]

        self._normalize_observations()

        # # For state change penalties
        # if previous_state != self.Meth_State:
        #     self.state_change = True
        # else:
        #     self.state_change = False

        reward = self._get_reward()
        observation = self.get_obs()
        terminated = self._is_terminated()

        self.k += 1

        state_c = {
            'i': self.i,
            'j': self.j,
            'k': self.k,
            'Meth_State': self.Meth_State,
            'Meth_T_cat': self.Meth_T_cat,
            'standby_op': self.standby_op,
            'startup_op': self.startup_op,
            'part_op': self.part_op,
            'full_op': self.full_op,
            'obs_norm': observation,
        }

        if self.train_or_eval == "train":
            info = self._get_info(state_c)
        else:
            info = self._get_info(state_c)

        # PtGEnv uses only "terminated" because preliminary studies showed no performance difference 
        # between using "terminated" and "truncated" episodes.

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)    # Reset the random seed

        global ep_index

        # Initialize dynamic variables for simulation and time tracking
        if isinstance(self.eps_ind, np.ndarray): # True for the training environment
            self.act_ep_h = int(self.eps_ind[ep_index] * self.eps_len_d * 24)
            self.act_ep_d = int(self.eps_ind[ep_index] * self.eps_len_d)
            ep_index += 1  # Choose next data subset for next episode
        else:               # For validation and test environments
            self.act_ep_h, self.act_ep_d = 0, 0
        self.clock_hours = 0 * self.time_step_size_sim / 3600  # in hours
        self.clock_days = self.clock_hours / 24  # in days

        self._initialize_datasets()
        self._initialize_op_rew()
        self._normalize_observations()

        observation = self.get_obs()

        state_c = {
            'i': self.i,
            'j': self.j,
            'k': self.k,
            'Meth_State': self.Meth_State,
            'Meth_T_cat': self.Meth_T_cat,
            'standby_op': self.standby_op,
            'startup_op': self.startup_op,
            'part_op': self.part_op,
            'full_op': self.full_op,
            'obs_norm': observation,
        }

        info = self._get_info(state_c)

        return observation, info

    def _is_terminated(self):
        """Returns whether the episode terminates"""
        if self.k == self.eps_sim_steps - 6:    return True     # Curtails training to ensure and data overhead (-6)
        else:                                   return False

    # ------------------ Utility/Helper Functions for Predicting Process Dynamics and State Changes --------------------------
    def _get_index(self, operation, t_cat):
        """
            Determine the position (index) in the operation data set based on the catalyst temperature
            :param operation: np.array of operation modes for each timestep
            :param t_cat: Catalyst temperature
            :return: idx: Index of the operation mode closest to the target temperature
        """
        diff = np.abs(operation[:, 1] - t_cat)      # Calculate temperature difference
        idx = diff.argmin()                         # Find the index with the smallest difference
        return idx

    def _perform_sim_step(self, operation, initial_state, next_operation, next_state, idx, j, change_operation):
        """
            Performs a single simulation step
            :param operation: np.array of operation modes for each timestep
            :param initial_state: The initial methanation state at the current timestep
            :param next_operation: np.array of the next operation mode (if change_operation == True)
            :param next_state: The final state after reaching the specified total_steps
            :param idx: Index of the closest operation mode to the catalyst temperature
            :param j: Index of the next timestep
            :param change_operation: A flag indicating whether the operation mode changes (True if it does)
            :return: op_range: Operation range; r_state: Methanation state; idx; j
        """
        total_steps = len(operation[:, 1])
        if (idx + j * self.step_size) < total_steps:
            r_state = initial_state
            op_range = operation[int(idx + (j - 1) * self.step_size):int(idx + j * self.step_size), :]
        else:
            r_state = next_state
            time_overhead = int(idx + j * self.step_size) - total_steps
            if time_overhead < self.step_size:
                # For the time overhead, fill op_range for the timestep with values (next operation/end of the data set)
                op_head = operation[int(idx + (j - 1) * self.step_size):, :]
                if change_operation:
                    idx = time_overhead
                    j = 0
                    op_overhead = next_operation[:idx, :]
                else:
                    op_overhead = np.ones((time_overhead, op_head.shape[1])) * operation[-1, :]
                op_range = np.concatenate((op_head, op_overhead), axis=0)
            else:
                # For the time overhead, fill op_range for the timestep with values at the end of the data set
                op_range = np.ones((self.step_size, operation.shape[1])) * operation[-1, :]
        return op_range, r_state, idx, j

    def _cont(self, operation, initial_state, next_operation, next_state, change_operation):
        """
            Perform a single simulation step in the current methanation state operation.
            :param operation: np.array of operation modes for each timestep
            :param initial_state: The initial methanation state at the current timestep
            :param next_operation: np.array of the next operation mode (if change_operation == True)
            :param next_state: The final state after reaching the specified total_steps
            :param change_operation: A flag indicating whether the operation mode changes (True if it does)
            :return: op_range: Operation range; r_state: Methanation state; idx; j
        """
        self.j += 1
        return self._perform_sim_step(operation, initial_state, next_operation, next_state, self.i, self.j, change_operation)

    def _standby(self):
        """
            Transition the system to the 'Standby' methanation state and perform a simulation step
            :return: op_range: Operation range; r_state: Methanation state; idx; j
        """
        self.Meth_State = self.M_state['standby']
        # Select the standby operation mode
        if self.Meth_T_cat <= self.t_cat_standby:
            self.standby = self.standby_up
            self.standby_op = 'standby_up'
        else:
            self.standby = self.standby_down
            self.standby_op = 'standby_down'
        # np.random.randint(low=-10, high=10) introduces randomness into the environment
        self.i = int(max(self._get_index(self.standby, self.Meth_T_cat) +
                         self.np_random.normal(0, self.noise, size=1)[0], 0))
        self.j = 1

        return self._perform_sim_step(self.standby, self.Meth_State, self.standby, self.Meth_State,
                                      self.i, self.j, False)

    def _cooldown(self):
        """
            Transition the system to the 'Cooldown' methanation state and perform a simulation step
            :return: op_range: Operation range; r_state: Methanation state; idx; j
        """
        self.Meth_State = self.M_state['cooldown']
        # Get index of the specific state according to T_cat
        self.i = int(max(self._get_index(self.cooldown, self.Meth_T_cat) +
                         self.np_random.normal(0, self.noise, size=1)[0], 0))
        self.j = 1

        return self._perform_sim_step(self.cooldown, self.Meth_State, self.cooldown, self.Meth_State,
                                      self.i, self.j, False)

    def _startup(self):
        """
            Transition the system to the 'Startup' methanation state and perform a simulation step
            :return: op_range: Operation range; r_state: Methanation state; idx; j
        """
        self.Meth_State = self.M_state['startup']
        self.partial = self.op1_start_p
        self.part_op = 'op1_start_p'
        self.full = self.op2_start_f
        self.full_op = 'op2_start_f'
        # Select the startup operation mode
        if self.hot_cold == 0:
            self.startup = self.startup_cold
            self.startup_op = 'startup_cold'
        else:  # self.hot_cold == 1
            self.startup = self.startup_hot
            self.startup_op = 'startup_hot'
        self.i = int(max(self._get_index(self.startup, self.Meth_T_cat) +
                         self.np_random.normal(0, self.noise, size=1)[0], 0))
        self.j = 1

        return self._perform_sim_step(self.startup, self.Meth_State, self.partial, self.M_state['partial_load'],
                                      self.i, self.j, True)

    def _partial(self):
        """
            Transition the system to the 'Partial load' state and perform a simulation step, dependent on prior full-load conditions.
            :return: op_range: Operation range; r_state: Methanation state; idx; j
        """
        self.Meth_State = self.M_state['partial_load']
        # Select the partial_load operation mode
        time_op = self.i + self.j * self.step_size  # Simulation step in full_load

        match self.full_op:
            case 'op2_start_f':
                if time_op < self.time2_start_f_p:
                    self.partial = self.op1_start_p     # Approximation: A simple change without considering temperature changes
                    self.part_op = 'op1_start_p'
                    self.i = self._get_index(self.partial, self.Meth_T_cat)
                    self.j = 1
                else:
                    self.partial = self.op8_f_p
                    self.part_op = 'op8_f_p'
                    self.i = 0
                    self.j = 1
            case 'op3_p_f':
                if time_op < self.time1_p_f_p:          # Approximation:  A simple return to the original state
                    self.partial = self.op8_f_p
                    self.part_op = 'op8_f_p'
                    self.i = self.i_fully_developed     # Fully developed operation
                    self.j = self.j_fully_developed
                    self.Meth_T_cat = self.op8_f_p[-1, 1]
                elif self.time1_p_f_p < time_op < self.time2_p_f_p:
                    self.partial = self.op4_p_f_p_5
                    self.part_op = 'op4_p_f_p_5'
                    self.j += 1
                elif self.time2_p_f_p < time_op < self.time_p_f:
                    self.partial = self.op4_p_f_p_5
                    self.part_op = 'op4_p_f_p_5'
                    self.i = self.time2_p_f_p
                    self.j = 1
                elif self.time_p_f < time_op < self.time34_p_f_p:
                    self.partial = self.op5_p_f_p_10
                    self.part_op = 'op5_p_f_p_10'
                    self.i = self.time3_p_f_p
                    self.j = 1
                elif self.time34_p_f_p < time_op < self.time45_p_f_p:
                    self.partial = self.op6_p_f_p_15
                    self.part_op = 'op6_p_f_p_15'
                    self.i = self.time4_p_f_p
                    self.j = 1
                elif self.time45_p_f_p < time_op < self.time5_p_f_p:
                    self.partial = self.op7_p_f_p_22
                    self.part_op = 'op7_p_f_p_22'
                    self.i = self.time5_p_f_p
                    self.j = 1
                else:  # time_op > self.time5_p_f_p
                    self.partial = self.op8_f_p
                    self.part_op = 'op8_f_p'
                    self.i = 0
                    self.j = 1
            case _ : # Full load operation: op9_f_p_f_5, op10_f_p_f_10, op11_f_p_f_15, op12_f_p_f_22
                self.partial = self.op8_f_p
                self.part_op = 'op8_f_p'
                self.i = 0
                self.j = 1

        return self._perform_sim_step(self.partial, self.Meth_State, self.partial, self.M_state['partial_load'],
                                      self.i, self.j, False)

    def _full(self):
        """
            Transition the system to the 'Full load' state and perform a simulation step, dependent on prior partial-load conditions.
            :return: op_range: Operation range; r_state: Methanation state; idx; j
        """
        self.Meth_State = self.M_state['full_load']
        # Select the full_load operation mode
        time_op = self.i + self.j * self.step_size  # Simulation step in partial_load

        match self.part_op:
            case 'op1_start_p':
                if time_op < self.time1_start_p_f:
                    self.full = self.op2_start_f    # Approximation: A simple change without considering temperature changes
                    self.full_op = 'op2_start_f'
                    self.i = 0
                    self.j = 1
                else:
                    self.full = self.op3_p_f
                    self.full_op = 'op3_p_f'
                    self.i = 0
                    self.j = 1
            case 'op8_f_p':
                if time_op < self.time1_f_p_f:      # Approximation: A simple return to the original state
                    self.full = self.op3_p_f
                    self.full_op = 'op3_p_f'
                    self.i = self.i_fully_developed # Fully developed operation
                    self.j = self.j_fully_developed
                    self.Meth_T_cat = self.op3_p_f[-1, 1]
                elif self.time1_f_p_f < time_op < self.time_f_p:
                    self.full = self.op9_f_p_f_5
                    self.full_op = 'op9_f_p_f_5'
                    self.j += 1
                elif self.time_f_p < time_op < self.time23_f_p_f:
                    self.full = self.op9_f_p_f_5
                    self.full_op = 'op9_f_p_f_5'
                    self.i = self.time2_f_p_f
                    self.j = 1
                elif self.time23_f_p_f < time_op < self.time34_f_p_f:
                    self.full = self.op10_f_p_f_10
                    self.full_op = 'op10_f_p_f_10'
                    self.i = self.time3_f_p_f
                    self.j = 1
                elif self.time34_f_p_f < time_op < self.time45_f_p_f:
                    self.full = self.op11_f_p_f_15
                    self.full_op = 'op11_f_p_f_15'
                    self.i = self.time4_f_p_f
                    self.j = 1
                elif self.time45_f_p_f < time_op < self.time5_f_p_f:
                    self.full = self.op12_f_p_f_20
                    self.full_op = 'op12_f_p_f_20'
                    self.i = self.time5_f_p_f
                    self.j = 1
                else:  # time_op > self.time5_f_p_f
                    self.full = self.op3_p_f
                    self.full_op = 'op3_p_f'
                    self.i = 0
                    self.j = 1
            case _:  # Partial load operation: op4_p_f_p_5, op5_p_f_p_10, op6_p_f_p_15, op7_f_p_f_22
                self.full = self.op3_p_f
                self.full_op = 'op3_p_f'
                self.i = 0
                self.j = 1

        return self._perform_sim_step(self.full, self.Meth_State, self.full, self.M_state['full_load'],
                                      self.i, self.j, False)
    
    def __deepcopy__(self, memo):
        """
            Custom deepcopy to avoid copying large static arrays.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # List of large static arrays to reference, not copy
        static_arrays = [
            'e_r_b', 'g_e',
            'startup_cold', 'startup_hot', 'cooldown',
            'standby_down', 'standby_up',
            'op1_start_p', 'op2_start_f', 'op3_p_f',
            'op4_p_f_p_5', 'op5_p_f_p_10', 'op6_p_f_p_15', 'op7_p_f_p_20',
            'op8_f_p', 'op9_f_p_f_5', 'op10_f_p_f_10', 'op11_f_p_f_15', 'op12_f_p_f_20'
        ]

        for k, v in self.__dict__.items():
            # Debug: print attribute name and shape if it's a numpy array
            # if isinstance(v, np.ndarray):
            #     print(f"Deepcopy: {k}, shape: {v.shape}, dtype: {v.dtype}")
            if k in static_arrays:
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

# ----------------------------------------------------------------------------------------------------------------------
print("Simulation...")
# env_meth = MethEnvSimulation()

timesteps = TOTAL_SIM_STEPS                # A day in 2 second steps
# timesteps = 20000
Meth_state = np.zeros((timesteps,))
T_cat = np.zeros((timesteps,))
Meth_H2_flow = np.zeros((timesteps,))
Meth_CH4_flow = np.zeros((timesteps,))
Meth_H2O_flow = np.zeros((timesteps,))
Meth_el_heating = np.zeros((timesteps,))
Meth_Actions = np.zeros((timesteps,))
Meth_Hot_Cold = np.zeros((timesteps,))
# states: standby=1, cooldown=2, startup=3, partial_load=4, full_load=5
# actions: standby=9, cooldown=6, startup=2, partial_load=10, full_load=11
# validation set
action = 6
actions = np.array([2, 11, 10, 11, 10, 11, 10, 11, 10, 6, 2, 6, 2, 11, 10, 9, 2, 11, 6, 2, 11, 10, 11, 10, 6])
action_steps = np.array([1, 3486, 5365, 5900, 7252, 7296, 7452, 7838, 8056, 9093, 9186, 9234, 11978, 20045, 20766,
                         23813, 36771, 42207, 43476, 46737, 48229, 56957, 57595, 57919, 62331])

for i in tqdm(range(timesteps), desc="Simulation"):
    # print("Prediction Step " + str(i))
    for k in range(len(actions)):
        if (i*SIM_STEP) >= np.around(action_steps[k]):
            action = actions[k]
    if i == action_steps[0]:
        action = 2
    Meth_Actions[i] = action
    env_meth.step(action)

# ----------------------------------------------------------------------------------------------------------------------
print("Plot results...")
time_sim = np.zeros((timesteps,))
time_val = np.zeros((timesteps+1,))
for i in range(timesteps):
    time_sim[i] = i * TIME_STEP_SIZE_SIM
    time_val[i] = i * TIME_STEP_SIZE_OP
time_val[-1] = timesteps * TIME_STEP_SIZE_OP

print(max(env_meth.Meth_CH4_flow_records))
print(min(env_meth.Meth_CH4_flow_records))

Meth_state = env_meth.Meth_State_records
T_cat = env_meth.Meth_T_cat_records
Meth_H2_flow = env_meth.Meth_H2_flow_records
Meth_H2_res_flow = env_meth.Meth_H2_res_flow_records
Meth_CH4_flow = env_meth.Meth_CH4_flow_records
Meth_H2O_flow = env_meth.Meth_H2O_flow_records
Meth_el_heating = env_meth.Meth_el_heating_records
Meth_Hot_Cold = env_meth.Meth_hot_cold_records

T_cat_val = np_validation[:, 1]
Meth_H2_flow_val = np_validation[:, 2]
Meth_CH4_flow_val = np_validation[:, 3]
Meth_H2_res_flow_val = np_validation[:, 4]
Meth_H2O_flow_val = np_validation[:, 5]
Meth_el_heating_val = np_validation[:, 6]

time_sim = time_sim/3600

time_val = time_val/3600

def calculate_RMSE(data_sim, data_val):
    mse = 0
    m = len(data_sim)
    # calculate MSE
    for i in range(m):
        mse += (data_val[i] - data_sim[i]) ** 2
        # print(mse, data_val[i], data_sim[i], (data_val[i] - data_sim[i]))

    mse = mse / m
    rmse = mse ** (0.5)
    # print(rmse, mse)

    return rmse

def calculate_MAE(data_sim, data_val):
    mae = 0
    m = len(data_sim)
    # calculate MSE
    for i in range(m):
        mae += abs(data_val[i] - data_sim[i])
        # print(mae, data_val[i], data_sim[i], (data_val[i] - data_sim[i]))

    mae = mae / m

    return mae

def calculate_MAPE(data_sim, data_val):
    mape = 0
    m = len(data_sim)
    m_corr = 0
    # calculate MSE
    for i in range(m):
        if data_val[i] != 0:
            mape += abs(data_val[i] - data_sim[i]) / data_val[i]
            m_corr += 1
        # print(mae, data_val[i], data_sim[i], (data_val[i] - data_sim[i]))

    mape = mape / m

    return mape*100

print("RMSE: T_cat - ", round(calculate_RMSE(T_cat+273.15, T_cat_val+273.15),1))
print("RMSE: Meth_H2_flow - ", round(calculate_RMSE(Meth_H2_flow, Meth_H2_flow_val)*1000,6))
print("RMSE: Meth_CH4_flow - ", round(calculate_RMSE(Meth_CH4_flow, Meth_CH4_flow_val)*1000,6))
print("RMSE: Meth_H2_res_flow - ", round(calculate_RMSE(Meth_H2_res_flow, Meth_H2_res_flow_val)*1000,6))
print("RMSE: Meth_H2O_flow - ", round(calculate_RMSE(Meth_H2O_flow, Meth_H2O_flow_val),6))
print("RMSE: Meth_el_heating - ", round(calculate_RMSE(Meth_el_heating, Meth_el_heating_val),0))

print("MAPE: T_cat - ", round(calculate_MAPE(T_cat+273.15, T_cat_val+273.15),1))
print("MAPE: Meth_H2_flow - ", round(calculate_MAPE(Meth_H2_flow, Meth_H2_flow_val),1))
print("MAPE: Meth_CH4_flow - ", round(calculate_MAPE(Meth_CH4_flow, Meth_CH4_flow_val),1))
print("MAPE: Meth_H2_res_flow - ", round(calculate_MAPE(Meth_H2_res_flow, Meth_H2_res_flow_val),1))
print("MAPE: Meth_H2O_flow - ", round(calculate_MAPE(Meth_H2O_flow, Meth_H2O_flow_val),1))
print("MAPE: Meth_el_heating - ", round(calculate_MAPE(Meth_el_heating, Meth_el_heating_val),1))



# fig, axs = plt.subplots(5, 1, figsize=(11, 7), sharex=True, sharey=False)
# # axs[0].plot(time_sim, Meth_state, label="Meth_State", marker='o', markersize=2)
# axs[0].plot(time_sim, Meth_state, 'k', label="Meth_State")
# # axs[0].plot(time_sim, Meth_Hot_Cold, color='g', label="Meth_Hot_Cold")
# # axs[0].set_ylim([0, 5.5])
# axs[0].set_ylabel('state s', rotation=0, labelpad=30)
# # axs[0].legend(loc="upper right",fontsize='small')
# axs[0].set_yticks([1,2,3,4,5])
# axs[0].set_yticklabels(['Standby', 'Cooldown/Off', 'Startup', 'Partial Load', 'Full Load'])
# # axs[1].set_ylim([0, 12])
# axs[0].set_ylabel(' ')
# axs[0].legend(loc="upper left", fontsize='small') #, bbox_to_anchor = (0.0, 0.0), ncol = 1, fancybox = True, shadow = True)
# axs[0].grid(axis='y', linestyle='dashed')
# # axs[1].plot(time_sim, Meth_Actions, label="Meth_Actions")
# # axs[1].set_ylim([0, 12])
# # axs[1].set_ylabel('a')
# axs[1].plot(time_sim, T_cat,'r', label="sim")
# axs[1].plot(time_val, T_cat_val, color='lightcoral', linestyle='--',label="exp")
# axs[1].set_ylim([0, 650])
# axs[1].set_ylabel('T$_{cat;max} $ [°C]', rotation=0, labelpad=40)
# axs[1].legend(loc="upper right",fontsize='small')
# axs[2].plot(time_sim, Meth_H2_flow*1000, 'b' , label="H2 (sim)")
# axs[2].plot(time_sim, Meth_CH4_flow, 'g', label="CH4 (sim)")
# axs[2].plot(time_val, Meth_H2_flow_val*1000, color='lightgray', linestyle='--', label="H2 (exp)")
# axs[2].plot(time_val, Meth_CH4_flow_val*1000, color='lightgreen', linestyle='--', label="CH4 (exp)")
# axs[2].set_ylim([0, 0.025*1000])
# axs[2].set_ylabel('molar flow [mmol/s]', rotation=0, labelpad=60)
# axs[2].legend(loc="upper right",fontsize='small')
# axs[3].plot(time_sim, Meth_H2O_flow, label="sim")
# axs[3].plot(time_val, Meth_H2O_flow_val, color='lightgray', linestyle='--',label="exp")
# axs[3].set_ylim([0, 0.72])
# axs[3].set_ylabel('steam [kg/h]', rotation=0, labelpad=50)
# axs[3].legend(loc="upper right",fontsize='small')
# axs[4].plot(time_sim, Meth_el_heating, label="sim")
# axs[4].plot(time_val, Meth_el_heating_val, color='lightgray', linestyle='--', alpha=0.7, label="exp")
# axs[4].set_ylim([-10, 2000])
# axs[4].set_xlabel('Time [h]')
# axs[4].set_xlim([0, 48])
# axs[4].set_ylabel('$P_{el}$ [W]', rotation=0, labelpad=25)
# axs[4].legend(loc="upper right",fontsize='small')
# plt.savefig("plots/Fig23.png", bbox_inches='tight', dpi=100)
#
# fig.suptitle('Simulation')
# plt.show()

plt.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['axes.linewidth'] = 1

# fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True, sharey=False)
# axs[0].plot(time_sim, Meth_H2O_flow, color='deepskyblue', linestyle='--',label="dif")
# axs[1].plot(time_sim, Meth_H2O_flow_val[:len(time_sim)], color='deepskyblue', linestyle='--',label="dif")
# axs[2].plot(time_sim, Meth_H2O_flow-Meth_H2O_flow_val[:len(time_sim)], color='deepskyblue', linestyle='--',label="dif")
#
# plt.show()


fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True, sharey=False)
axs[0].grid()
axs[0].plot(time_sim, Meth_state, 'k', label='$\Omega_{MT}$', linewidth=2)
axs[0].set_yticks([1,2,3,4,5])
axs[0].set_yticklabels(['Standby', 'Coold.', 'Startup', 'Partial', 'Full'])
# axs[1].set_ylim([0, 12])
axs[0].set_ylabel(' ')
legend0 = axs[0].legend(bbox_to_anchor=(0.8, 0.55, 0.2, 1), loc=3, ncol=1, mode="expand", framealpha=1, prop={'size': 12})
frame = legend0.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
frame.set_boxstyle('Square', pad=0.2)
axs[1].grid()
axs[1].plot(time_sim, T_cat+273.15,'b', label="Sim")
axs[1].plot(time_val, T_cat_val+273.15, color='deepskyblue', linestyle='--',label="Exp")
axs[1].set_ylabel('T$_{cat}$ [K]', rotation=90, labelpad=5)
axs[2].set_yticks([400, 600, 800])
axs[2].set_ylim([270, 850])
legend1 = axs[1].legend(bbox_to_anchor=(0.8, 0.35, 0.2, 1), loc=3, ncol=1, mode="expand", framealpha=1, prop={'size': 12})
frame = legend1.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
frame.set_boxstyle('Square', pad=0.2)
# ax2_1 = axs[1].twinx()
# ax2_1.plot(time_sim, Meth_el_heating, 'k',label="sim", alpha=0.8, linewidth=2)
# ax2_1.plot(time_val, Meth_el_heating_val, color='lightgray', linestyle='--', alpha=0.7, label="exp")
# ax2_1.set_ylim([-10, 2000])
# ax2_1.set_xlabel('Time [h]')
# ax2_1.set_xlim([0, 48])
# ax2_1.set_ylabel('P$_{Meth,el}$ [W]', rotation=90, labelpad=0)
# ax2_1.legend(loc="upper right",fontsize='small')


axs[2].grid()
axs[2].plot(time_sim, Meth_H2_flow*1000, 'cadetblue' , label="H$_{2}$")
axs[2].plot(time_sim, Meth_CH4_flow*1000, 'green', label="CH$_{4,syn}$")
axs[2].plot(time_val, Meth_H2_flow_val*1000, color='lightblue', linestyle='--')
axs[2].plot(time_val, Meth_CH4_flow_val*1000, color='lightgreen', linestyle='--')
axs[2].set_ylim([0, 0.025*1000])
axs[2].set_yticks([0,10, 20])
axs[2].set_ylabel('$\dot{n}_i$ [mmol/s]', rotation=90, labelpad=0)
axs[2].legend(loc="upper right",fontsize='small', ncol = 2)
legend2 = axs[2].legend(bbox_to_anchor=(0.8, 0.25, 0.2, 1), loc=3, ncol=1, mode="expand", framealpha=1, prop={'size': 12})
frame = legend2.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
frame.set_boxstyle('Square', pad=0.2)


axs[3].grid()
axs[3].plot(time_sim, Meth_el_heating, 'k',label='P$_{MT,el}$')
axs[3].plot(time_val, Meth_el_heating_val, color='lightgray', linestyle='--', alpha=0.7)
axs[3].set_ylim([-10, 2000])
axs[3].set_yticks([0,1000, 2000])
axs[3].set_xlabel('Time [h]')
axs[3].set_xlim([0, 48])
axs[3].set_ylabel('P$_{MT,el}$ [W]', rotation=90, labelpad=0)
ax2_1 = axs[3].twinx()
ax2_1.set_ylim([0, 1])
ax2_1.plot(time_sim, Meth_H2O_flow, 'b', label='$\dot{m}_{St}$')
ax2_1.plot(time_val, Meth_H2O_flow_val, 'deepskyblue', linestyle='--', alpha=0.7)
ax2_1.set_ylabel('$\dot{m}_{St}$ [kg/h]')

h1, l1 = axs[3].get_legend_handles_labels()
h2, l2 = ax2_1.get_legend_handles_labels()
legend3 = axs[3].legend(h1+h2, l1+l2, bbox_to_anchor=(0.8, 0.24, 0.2, 1), loc=3, ncol=1, mode="expand", framealpha=1,
                       prop={'size': 12})
frame = legend3.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
frame.set_boxstyle('Square', pad=0.2)



box = axs[0].get_position()
axs[0].set_position([box.x0-0.032, box.y0+0.07, box.width+0.055, box.height+0.04])
box = axs[1].get_position()
axs[1].set_position([box.x0-0.032, box.y0+0.04, box.width+0.055, box.height+0.04])
box = axs[2].get_position()
axs[2].set_position([box.x0-0.032, box.y0+0.01, box.width+0.055, box.height+0.04])
box = axs[3].get_position()
axs[3].set_position([box.x0-0.032, box.y0-0.02, box.width+0.055, box.height+0.04])

# fig.suptitle(" Alg:" + plot_name + "\n Rew:" + str(np.round(stats_dict['Meth_cum_reward_stats'][-1], 0)))
plt.savefig('plots/Val_5kw.pdf')
# print("Reward =", stats_dict['Meth_cum_reward_stats'][-1])

# plt.close()
plt.show()




