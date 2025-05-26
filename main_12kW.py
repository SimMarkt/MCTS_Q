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

from numba.experimental import jitclass
from numba import float64, boolean, int32, int64, types, typeof
import numpy as np
# import gym
# from collections import OrderedDict
# from stable_baselines3.common.env_checker import check_env
# from gym.wrappers.time_limit import TimeLimit
# from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines.common.callbacks import CheckpointCallback

# from stable_baselines3 import PPO

# Import modules and utils
from utils import import_historic_prices, import_data, df_to_nparray

# ---------------------------------------------------------------------------------------------------------------------
print("Specify model parameters and classes...")  ### ToDO: PROVE VALUES BEFORE FINAL TRAINING AGAIN
class GlobalParams:
    def __init__(self):
        self.datafile_path1 = "data/data-day-ahead.csv"
        self.datafile_path2 = "data/data-meth_startup_cold.csv"
        self.datafile_path3 = "data/data-meth_startup_hot.csv"
        self.datafile_path4 = "data/data-meth_cooldown.csv"
        self.datafile_path5 = "data/data-meth_standby_down.csv"      # from operation to Hot-Standby
        self.datafile_path6 = "data/data-meth_standby_up.csv"      # from shutdown to Hot-Standby
        # self.datafile_path7 = "data/data-meth_op1_start_p.csv"
        # self.datafile_path8 = "data/data-meth_op2_start_f.csv"
        # self.datafile_path9 = "data/data-meth_op3_p_f.csv"
        # self.datafile_path10 = "data/data-meth_op4_p_f_p_5.csv"
        # self.datafile_path11 = "data/data-meth_op5_p_f_p_10.csv"
        # self.datafile_path12 = "data/data-meth_op6_p_f_p_15.csv"
        # self.datafile_path13 = "data/data-meth_op7_p_f_p_22.csv"
        # self.datafile_path14 = "data/data-meth_op8_f_p.csv"
        # self.datafile_path15 = "data/data-meth_op9_f_p_f_5.csv"
        # self.datafile_path16 = "data/data-meth_op10_f_p_f_10.csv"
        # self.datafile_path17 = "data/data-meth_op11_f_p_f_15.csv"
        # self.datafile_path18 = "data/data-meth_op12_f_p_f_20.csv"
        self.datafile_path7 = "data/data-meth_op1_start_p_12kw.csv"
        self.datafile_path8 = "data/data-meth_op2_start_f_12kw.csv"
        self.datafile_path9 = "data/data-meth_op3_p_f_12kw.csv"
        self.datafile_path10 = "data/data-meth_op4_p_f_p_5_12kw.csv"
        self.datafile_path11 = "data/data-meth_op5_p_f_p_10_12kw.csv"
        self.datafile_path12 = "data/data-meth_op6_p_f_p_15_12kw.csv"
        self.datafile_path13 = "data/data-meth_op7_p_f_p_22_12kw.csv"
        self.datafile_path14 = "data/data-meth_op8_f_p_12kw.csv"
        self.datafile_path15 = "data/data-meth_op9_f_p_f_5_12kw.csv"
        self.datafile_path16 = "data/data-meth_op10_f_p_f_10_12kw.csv"
        self.datafile_path17 = "data/data-meth_op11_f_p_f_15_12kw.csv"
        self.datafile_path18 = "data/data-meth_op12_f_p_f_20_12kw.csv"
        self.datafile_path19 = "data/data_meth_validation_12kW.csv"
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
TOTAL_SIM_STEPS = 104833



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
        self.Meth_CH4_flow_records = np.empty((TOTAL_SIM_STEPS))
        self.Meth_H2O_flow_records = np.empty((TOTAL_SIM_STEPS))
        self.Meth_el_heating_records = np.empty((TOTAL_SIM_STEPS))
        self.Meth_hot_cold_records = np.empty((TOTAL_SIM_STEPS))
        self.Meth_H2_res_flow_records = np.empty((TOTAL_SIM_STEPS))

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
        self.Meth_CH4_flow_records[k] = self.Meth_CH4_flow
        self.Meth_H2O_flow_records[k] = self.Meth_H2O_flow
        self.Meth_el_heating_records[k] = self.Meth_el_heating
        self.Meth_hot_cold_records[k] = self.hot_cold
        self.Meth_H2_res_flow_records[k] = self.Meth_H2_res_flow

        self.k += 1


# ----------------------------------------------------------------------------------------------------------------------
print("Simulation...")
env_meth = MethEnvSimulation()

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
actions = np.array([2, 11, 10, 11, 10, 6, 2, 11, 10, 11, 10, 11, 10, 11, 9, 2, 11, 10, 11, 10, 9, 6])
action_steps = np.array([272, 4663, 12588, 13470, 18185, 19297, 34853, 39796, 43001, 43780, 44170, 44406, 47471, 47909,
                         50612, 52969, 55706, 55806, 59002, 59462, 62129, 76319])

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
Meth_CH4_flow = env_meth.Meth_CH4_flow_records
Meth_H2O_flow = env_meth.Meth_H2O_flow_records
Meth_H2_res_flow = env_meth.Meth_H2_res_flow_records
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
axs[2].set_ylim([0, 0.060*1000])
axs[2].set_yticks([0, 20, 40, 60])
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
ax2_1.set_ylim([0, 2.5])
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
plt.savefig('plots/Val_12kw.pdf')
# print("Reward =", stats_dict['Meth_cum_reward_stats'][-1])

# plt.close()
plt.show()



