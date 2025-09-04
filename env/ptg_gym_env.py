"""
Custom Environment implementing the Gymnasium interface for PtG dispatch optimization.
Version 1.0
Modifications: Observation space with process time-series data of the last time step

Abbreviations:
   SNG: Synthetic natural gas
   EUA: European emission allowances
   CHP: Combined heat and power plant
   CH4: Methane
   H2: Hydrogen
   O2: Oxygen
   CO2: Carbon dioxide
   H2O_DE: Water vapor (steam)
   LHV: Lower heating value
   EEG: German Renewable Energy Act (Erneuerbare-Energien-Gesetz)

Optional modifications: Include Meth_State and/or Action history in the observation space
Optional modifications: Include temporal encoding in the observation space 
                        (sin/cos for minute within hour and hour within day)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import copy

ep_index = 0

class PTGEnv(gym.Env):
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
        self.startup = self.startup_cold                            # Current startup data set
        self.partial = self.op1_start_p                             # Current partial load data set
        self.part_op = 'op1_start_p'                                # Track partial load conditions
        self.full = self.op2_start_f                                # Current full load data set
        self.full_op = 'op2_start_f'                                # Track full load conditions
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

    def _get_info(self):
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

    def step(self, action):
        k = self.k

        if self.Meth_T_cat <= self.t_cat_startup_cold:
            self.hot_cold = 0
        elif self.Meth_T_cat >= self.t_cat_startup_hot:
            self.hot_cold = 1

        previous_state = self.Meth_State

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

        # For state change penalties
        if previous_state != self.Meth_State:
            self.state_change = True
        else:
            self.state_change = False

        reward = self._get_reward()
        observation = self.get_obs()
        terminated = self._is_terminated()
        if self.train_or_eval == "train":
            info = {}
        else:
            info = self._get_info()

        self.k += 1

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
        info = self._get_info()

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
        else:
            self.standby = self.standby_down
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
        else:  # self.hot_cold == 1
            self.startup = self.startup_hot
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