# ----------------------------------------------------------------------------------------------------------------
# MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
# GitHub Repository: https://github.com/SimMarkt/MCTS_Q
#
# mctsq_utils: 
# > Utiliy/Helper functions
# ----------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import gymnasium as gym 

from src.mctsq_opt import calculate_optimum

def import_market_data(csvfile: str, type: str, path: str):
    """
        Imports day-ahead market price data.
        :param csvfile: Name of the .csv file containing market data ["Time [s]"; <data>].
        :param type: Type of market data to be imported ('el' for electricity, 'gas' for natural gas, 'eua' for EUA prices).
        :param path: File path to the RL_PtG project directory.
        :return: A numpy array containing the extracted market data.
    """

    file_path = path + "/" + csvfile
    df = pd.read_csv(file_path, delimiter=";", decimal=".")
    df["Time"] = pd.to_datetime(df["Time"], format="%d-%m-%Y %H:%M")

    if type == "el":        # Electricity price data
        arr = df["Day-Ahead-price [Euro/MWh]"].values.astype(float) / 10        # Convert Euro/MWh into ct/kWh
    elif type == "gas":     # Gas price data
        arr = df["THE_DA_Gas [Euro/MWh]"].values.astype(float) / 10             # Convert Euro/MWh into ct/kWh
    elif type == "eua":     # EUA price data
        arr = df["EUA_CO2 [Euro/t]"].values.astype(float)
    else:
        assert False, "Invalid market data type. Must be one of ['el', 'gas', 'eua']!"

    return arr


def import_data(csvfile: str, path: str):
    """
        Imports experimental methanation process data.
        :param csvfile: Name of the .csv file containing operational data.
        :param path: File path to the RL_PtG project directory.
        :return: A numpy array containing time-series operational data.
    """
    # Import historic Data from csv file
    file_path = path + "/" + csvfile
    df = pd.read_csv(file_path, delimiter=";", decimal=".")

    # Transform data to numpy array
    t = df["Time [s]"].values.astype(float)                         # Time
    t_cat = df["T_cat [gradC]"].values.astype(float)                # Maximum catalyst temperature [°C]
    n_h2 = df["n_h2 [mol/s]"].values.astype(float)                  # Hydrogen reactant molar flow rate [mol/s]
    n_ch4 = df["n_ch4 [mol/s]"].values.astype(float)                # Methane product molar flow rate [mol/s]
    n_h2_res = df["n_h2_res [mol/s]"].values.astype(float)          # Hydrogen product molar flow rate (residues) [mol/s]    
    m_h2o = df["m_DE [kg/h]"].values.astype(float)                  # Water mass flow rate [kg/h]
    p_el = df["Pel [W]"].values.astype(float)                       # Electrical power consumption for heating the methanation plant [W]
    arr = np.c_[t, t_cat, n_h2, n_ch4, n_h2_res, m_h2o, p_el]

    return arr


def load_data(EnvConfig, TrainConfig):
    """
        Loads historical market data and experimental methanation operation data
        :param TrainConfig: Training configuration (class object)
        :return dict_price_data: Dictionary containing electricity, gas, and EUA market data
                dict_op_data: Dictionary containing time-series data for dynamic methanation operations
    """
    print("---Load data")
    path = TrainConfig.path

    # Define market data file types (electricity, gas, and EUA) and dataset splits.
    market_data_files = [('el_price', 'el'), ('gas_price', 'gas'), ('eua_price', 'eua')]
    splits = ['train', 'val', 'test']

    dict_price_data = {
        f'{data[0]}_{split}': import_market_data(getattr(EnvConfig, f'datafile_path_{split}_{data[1]}'), data[1], path)
        for data in market_data_files for split in splits
    }

    # Function to verify consistency of market data sizes across datasets.
    def check_market_data_size(split):
        el_size_h = len(dict_price_data[f'el_price_{split}'])
        el_size_d = el_size_h // 24
        gas_size_d = len(dict_price_data[f'gas_price_{split}'])
        eua_size_d = len(dict_price_data[f'eua_price_{split}'])

        if len({el_size_d, gas_size_d, eua_size_d}) > 1:
            warnings.warn(
                f"Market data size does not match for {split}: electricity ({el_size_h}h = {el_size_d}d), "
                f"gas ({gas_size_d}d), and EUA ({eua_size_d}d)! -> Check size!",
                UserWarning
            )

    # Apply the function to all splits
    for split in splits:
        check_market_data_size(split)

    # Define methanation operation datasets with corresponding file indices
    op_data_files = [
        ('startup_cold', 2), ('startup_hot', 3), ('cooldown', 4), ('standby_down', 5), ('standby_up', 6),
        ('op1_start_p', 7), ('op2_start_f', 8), ('op3_p_f', 9), ('op4_p_f_p_5', 10), ('op5_p_f_p_10', 11),
        ('op6_p_f_p_15', 12), ('op7_p_f_p_22', 13), ('op8_f_p', 14), ('op9_f_p_f_5', 15), ('op10_f_p_f_10', 16),
        ('op11_f_p_f_15', 17), ('op12_f_p_f_20', 18)
    ]
    dict_op_data = {
        key: import_data(getattr(EnvConfig, f'datafile_path{num}'), path)
        for key, num in op_data_files
    }

    if EnvConfig.scenario in [2, 3]:  
        gas_price = EnvConfig.ch4_price_fix if EnvConfig.scenario == 2 else 0
        for key in ['gas_price_train', 'gas_price_val', 'gas_price_test']:
            dict_price_data[key] = np.full(len(dict_price_data[key]), gas_price)

        if EnvConfig.scenario == 3:  
            for key in ['eua_price_train', 'eua_price_val', 'eua_price_test']:
                dict_price_data[key] = np.zeros(len(dict_price_data[key]))

    # Set reward level values
    dict_price_data.update({f'{key}_reward_level': EnvConfig.r_0_values[key] 
                            for key in ['el_price', 'gas_price', 'eua_price']})

    # Check if training set is divisible by the episode length
    min_train_len = 6                       # Minimum No. of days in the training set 
    # RL training excludes 'min_train_len' days to maintain space for the day-ahead market price forecast.
    EnvConfig.train_len_d = len(dict_price_data['gas_price_train']) - min_train_len 
    assert EnvConfig.train_len_d > 0, f'The training set size must be greater than {min_train_len} days'
    if EnvConfig.train_len_d % EnvConfig.eps_len_d != 0:
        # Find all possible divisors of EnvConfig.train_len_d
        divisors = [i for i in range(1, EnvConfig.train_len_d + 1) if EnvConfig.train_len_d % i == 0]
        assert False, f'The training set size {EnvConfig.train_len_d} must be divisible by the episode length - data/config_env.yaml -> eps_len_d : {EnvConfig.eps_len_d}; Possible divisors are: {divisors}'
    # Ensure that the training set is larger or at least equal to the defined episode length  
    assert EnvConfig.train_len_d >= EnvConfig.eps_len_d, f'Training set size ({EnvConfig.train_len_d}) must be larger or at least equal to the defined episode length ({EnvConfig.eps_len_d})!'

    return dict_price_data, dict_op_data

class Preprocessing():
    """A class for preprocessing energy market and process data"""

    def __init__(self, dict_price_data, dict_op_data, EnvConfig, TrainConfig):
        """
            Initializes preprocessing with configuration parameters and data
            :param dict_price_data: Dictionary containing historical market data.
            :param dict_op_data: Dictionary containing dynamic process data.
            :param EnvConfig: Configuration settings for the environment.
            :param TrainConfig: Configuration settings for training.
        """
        # Store configuration objects and input data
        self.EnvConfig = EnvConfig
        self.TrainConfig = TrainConfig
        self.dict_price_data = dict_price_data
        self.dict_op_data = dict_op_data
        # Placeholder variables for computed values.
        self.dict_pot_r_b = None                    # Dictionary containing the potential reward [pot_rew...] and the boolean load identifier [part_full_b...]
        self.r_level = None                         # Defines the reward penalty level based on electricity, synthetic natural gas (SNG), and EUA price levels

        # e_r_b_train/e_r_b_val/e_r_b_test: (hourly values)
        #   np.array which stores elec. price data, potential reward, and boolean identifier
        #   Dimensions = [Type of data] x [No. of day-ahead values] x [historical values]
        #       Type of data = [el_price, pot_rew, part_full_b]
        #       No. of day-ahead values = EnvConfig.price_ahead + EnvConfig.price_past
        #       historical values = No. of values in the electricity price data set
        self.e_r_b_train, self.e_r_b_val, self.e_r_b_test = None, None, None

        # g_e_train/g_e_val/g_e_test: (daily values)
        #   np.array which stores gas and EUA price data
        #   Dimensions = [Type of data] x [No. of day-ahead values] x [historical values]
        #       Type of data = [gas_price, pot_rew, part_full_b]
        #       No. of day-ahead values = 2 (today and tomorrow)
        #       historical values = No. of values in the gas/EUA price data set
        self.g_e_train, self.g_e_val, self.g_e_test = None, None, None

        # Variables for splitting the training set into randomly selected subsets for episodic learning
        self.eps_sim_steps_train = None         # No. of steps per episode in the training set
        self.eps_sim_steps_val = None           # No. of steps in the validation set
        self.eps_sim_steps_test = None          # No. of steps in the test set
        self.eps_ind = None                     # Indexes of the randomly ordered training subsets
        self.overhead_factor = 10               # Overhead factor for self.eps_ind to account for randomness in process selection during multiprocessing (Must be an integer)
        assert isinstance(self.overhead_factor, int), f"Episode overhead ({self.overhead_factor}) must be an integer!"
        self.n_eps = None                       # Episode length in seconds
        self.num_loops = None                   # No. of loops over the full training set during training

        # For Multiprocessing: self.n_eps_loops is used for definition of different eps_ind for different processes (RL_PtG\env\ptg_gym_env.py for reference)
        self.n_eps_loops = None                 # Total No. of episodes across the entire training process 

        self.preprocessing_rew()                # Compute potential reward and load identifier
        self.preprocessing_array()              # Convert day-ahead datasets into NumPy arrays for calculations.
        self.define_episodes()                  # Define episodes and generate indices for random subset selection in training                

    def preprocessing_rew(self):
        """
            Data preprocessing, including the calculation of potential rewards.
            These represent the maximum possible reward in Power-to-Gas (PtG) operation, 
            either in partial load [part_full_b... = 0] or full load [part_full_b... = 1].
        """

        # Compute PtG operation data for the theoretical optimum (T-OPT), ignoring system dynamics.
        # The function calculate_optimum() was moved from Preprocessing() to the separate rl_opt.py file for better clarity.
        print("---Calculate the theoretical optimum, the potential reward, and the load identifier")
        stats_dict_opt_train = calculate_optimum(self.dict_price_data['el_price_train'], self.dict_price_data['gas_price_train'],
                                                self.dict_price_data['eua_price_train'], "Training", self.EnvConfig.stats_names)
        stats_dict_opt_val = calculate_optimum(self.dict_price_data['el_price_val'], self.dict_price_data['gas_price_val'],
                                                self.dict_price_data['eua_price_val'], "Validation", self.EnvConfig.stats_names)
        stats_dict_opt_test = calculate_optimum(self.dict_price_data['el_price_test'], self.dict_price_data['gas_price_test'],
                                                self.dict_price_data['eua_price_test'], "Test", self.EnvConfig.stats_names)
        stats_dict_opt_level = calculate_optimum(self.dict_price_data['el_price_reward_level'], self.dict_price_data['gas_price_reward_level'],
                                                self.dict_price_data['eua_price_reward_level'], "reward_Level", self.EnvConfig.stats_names)

        # Store datasets containing future values of potential rewards at two different load levels,
        # Additionally, store datasets for the load identifier corresponding to future potential rewards.
        # The part_full_b_... variable is determined as follows:
        #       if pot_rew_... <= 0:
        #           part_full_b_... = -1        # Operation is not profitable
        #       else:
        #           if (pot_rew_... in full load) < (pot_rew... in partial load):
        #               part_full_b_... = 0     # Partial load is most profitable
        #           else:
        #               part_full_b_... = 1     # Full load is most profitable
        self.dict_pot_r_b = {
            'pot_rew_train': stats_dict_opt_train['Meth_reward_stats'],
            'part_full_b_train': stats_dict_opt_train['part_full_stats'],
            'pot_rew_val': stats_dict_opt_val['Meth_reward_stats'],
            'part_full_b_val': stats_dict_opt_val['part_full_stats'],
            'pot_rew_test': stats_dict_opt_test['Meth_reward_stats'],
            'part_full_b_test': stats_dict_opt_test['part_full_stats'],
        }

        self.r_level = stats_dict_opt_level['Meth_reward_stats']


    def preprocessing_array(self):
        """Convert dictionaries to NumPy arrays for computational efficiency"""

        total_len = self.EnvConfig.price_ahead + self.EnvConfig.price_past

        # e_r_b: Multi-Dimensional array which stores Day-ahead electricity price data as well as Day-ahead potential rewards
        # and load identifiers for the entire training and test sets
        # e.g. e_r_b_train[0, 12, 156] represents the current value of the electricity price [0,-,-] at the
        # 156ths entry of the electricity price data set 
        self.e_r_b_train = np.zeros((3, total_len, self.dict_price_data['el_price_train'].shape[0] - total_len))
        self.e_r_b_val = np.zeros((3, total_len, self.dict_price_data['el_price_val'].shape[0] - total_len))
        self.e_r_b_test = np.zeros((3, total_len, self.dict_price_data['el_price_test'].shape[0] - total_len))

        for i in range(total_len):    
            self.e_r_b_train[0, i, :] = self.dict_price_data['el_price_train'][i:(-total_len + i)]
            self.e_r_b_train[1, i, :] = self.dict_pot_r_b['pot_rew_train'][i:(-total_len + i)]
            self.e_r_b_train[2, i, :] = self.dict_pot_r_b['part_full_b_train'][i:(-total_len + i)]
            self.e_r_b_val[0, i, :] = self.dict_price_data['el_price_val'][i:(-total_len + i)]
            self.e_r_b_val[1, i, :] = self.dict_pot_r_b['pot_rew_val'][i:(-total_len + i)]
            self.e_r_b_val[2, i, :] = self.dict_pot_r_b['part_full_b_val'][i:(-total_len + i)]
            self.e_r_b_test[0, i, :] = self.dict_price_data['el_price_test'][i:(-total_len + i)]
            self.e_r_b_test[1, i, :] = self.dict_pot_r_b['pot_rew_test'][i:(-total_len + i)]
            self.e_r_b_test[2, i, :] = self.dict_pot_r_b['part_full_b_test'][i:(-total_len + i)]

        # g_e: Multi-Dimensional Array which stores Day-ahead gas and EUA price data for the entire training and test set        
        self.g_e_train = np.zeros((2, 2, self.dict_price_data['gas_price_train'].shape[0]-1))
        self.g_e_val = np.zeros((2, 2, self.dict_price_data['gas_price_val'].shape[0]-1))
        self.g_e_test = np.zeros((2, 2, self.dict_price_data['gas_price_test'].shape[0]-1))

        self.g_e_train[0, 0, :] = self.dict_price_data['gas_price_train'][:-1]  
        self.g_e_train[1, 0, :] = self.dict_price_data['eua_price_train'][:-1]
        self.g_e_val[0, 0, :] = self.dict_price_data['gas_price_val'][:-1]
        self.g_e_val[1, 0, :] = self.dict_price_data['eua_price_val'][:-1]
        self.g_e_test[0, 0, :] = self.dict_price_data['gas_price_test'][:-1]
        self.g_e_test[1, 0, :] = self.dict_price_data['eua_price_test'][:-1]
        self.g_e_train[0, 1, :] = self.dict_price_data['gas_price_train'][1:]
        self.g_e_train[1, 1, :] = self.dict_price_data['eua_price_train'][1:]
        self.g_e_val[0, 1, :] = self.dict_price_data['gas_price_val'][1:]
        self.g_e_val[1, 1, :] = self.dict_price_data['eua_price_val'][1:]
        self.g_e_test[0, 1, :] = self.dict_price_data['gas_price_test'][1:]
        self.g_e_test[1, 1, :] = self.dict_price_data['eua_price_test'][1:]

    def define_episodes(self):
        """Defines settings for training and evaluation episodes"""

        print("---Define episodes and step size limits")
        # No. of days in the test set ("-1" accounts for the day-ahead overhead)
        val_len_d = len(self.dict_price_data['gas_price_val']) - 1
        test_len_d = len(self.dict_price_data['gas_price_test']) - 1

        # Divide the full training dataset into smaller subsets, each representing a separate episode
        self.n_eps = int(self.EnvConfig.train_len_d / self.EnvConfig.eps_len_d)  # No. of training subsets (episodes)
        self.eps_len = 24 * 3600 * self.EnvConfig.eps_len_d  # Episode length in [s]

        # No. of steps per episode in training, validation, and testing
        self.eps_sim_steps_train = int(self.eps_len / self.EnvConfig.sim_step)
        self.eps_sim_steps_val = int(24 * 3600 * val_len_d / self.EnvConfig.sim_step)
        self.eps_sim_steps_test = int(24 * 3600 * test_len_d /self.EnvConfig.sim_step)

        # Define the total number of loops over the entire training set for all workers
        self.num_loops = self.TrainConfig.train_steps / (self.eps_sim_steps_train * self.n_eps)            
        # Print training process details
        print("    > Total number of training steps =", self.TrainConfig.train_steps)
        print("    > No. of loops over the entire training set =", round(self.num_loops,3))
        print("    > Training steps per episode =", self.eps_sim_steps_train)
        print("    > Steps in the evaluation set =", self.eps_sim_steps_test, "\n")

        # Generate a randomized selection routine without replacement for different training subsets
        self.rand_eps_ind()

        # For multiprocessing, ensure eps_ind is not shared across processes
        self.n_eps_loops = self.n_eps * int(self.num_loops)  # Allows unique eps_ind assignments per process (RL_PtG\env\ptg_gym_env.py for reference)


    def rand_eps_ind(self):
        """
            The agent can either:
            1. Use the entire training set in a single episode (train_len_d == eps_len_d).
            2. Divide the training set into smaller subsets (train_len_d > eps_len_d), selecting subsets randomly.
        """

        np.random.seed(self.TrainConfig.seed_train)     # Set the random seed for random episode selection

        if self.EnvConfig.train_len_d == self.EnvConfig.eps_len_d:
            self.eps_ind = np.zeros(self.n_eps*int(self.num_loops)*self.overhead_factor)
        else:           # self.EnvConfig.train_len_d > self.EnvConfig.eps_len_d:
            # Random selection with replacement
            num_ep = np.linspace(start=0, stop=self.n_eps-1, num=self.n_eps)
            if self.num_loops < 1:      num_loops_int = 1
            else:                       num_loops_int = int(self.num_loops)
            random_ep = np.zeros((num_loops_int*self.overhead_factor, self.n_eps))
            for i in range(num_loops_int*self.overhead_factor):
                random_ep[i, :] = num_ep
                np.random.shuffle(random_ep[i, :])
            self.eps_ind = random_ep.reshape(int(self.n_eps*num_loops_int*self.overhead_factor)).astype(int)

    def dict_env_kwargs(self, type="train"):
        """
            Returns global model parameters and hyperparameters for the PtG environment as a dictionary.
            :param type: Specifies whether the dataset is for training ("train") or for validation/testing ("val_test").
            :return: env_kwargs - Dictionary containing global parameters and hyperparameters.
        """

        # General environment configurations
        env_kwargs = {
            **{f"ptg_{key}": self.EnvConfig.ptg_state_space[key] for key in 
            ["standby", "cooldown", "startup", "partial_load", "full_load"]},
            **{key: getattr(self.EnvConfig, key) for key in 
            ["noise", "eps_len_d", "sim_step", "time_step_op", "price_ahead", "price_past", "scenario",
                "convert_mol_to_Nm3", "H_u_CH4", "H_u_H2", "dt_water", "cp_water", "rho_water",
                "Molar_mass_CO2", "Molar_mass_H2O", "h_H2O_evap", "eeg_el_price", "heat_price",
                "o2_price", "water_price", "min_load_electrolyzer", "max_h2_volumeflow", "eta_CHP",
                "t_cat_standby", "t_cat_startup_cold", "t_cat_startup_hot", "time1_start_p_f",
                "time2_start_f_p", "time_p_f", "time_f_p", "time1_p_f_p", "time2_p_f_p",
                "time23_p_f_p", "time3_p_f_p", "time34_p_f_p", "time4_p_f_p", "time45_p_f_p",
                "time5_p_f_p", "time1_f_p_f", "time2_f_p_f", "time23_f_p_f", "time3_f_p_f",
                "time34_f_p_f", "time4_f_p_f", "time45_f_p_f", "time5_f_p_f", "i_fully_developed", 
                "j_fully_developed", "el_l_b", "el_u_b", "gas_l_b", "gas_u_b", "eua_l_b", "eua_u_b",
                "T_l_b", "T_u_b", "h2_l_b", "h2_u_b", "ch4_l_b", "ch4_u_b", "h2_res_l_b", "h2_res_u_b",
                "h2o_l_b", "h2o_u_b", "heat_l_b", "heat_u_b"]},
            "n_eps_loops": self.n_eps_loops,
            "reward_level": self.r_level,
            "action_type": "discrete",
        }

        # Operation data
        env_kwargs.update({key: self.dict_op_data[key] for key in self.dict_op_data})
        env_kwargs.update({
                "seq_length": self.TrainConfig.seq_length,              # Length of the sequence for time-series encoding
                "seq_step": self.TrainConfig.seq_step,                  # Number of data steps in process data between two consequitive entries in the sequences
                "state_change_penalty": self.EnvConfig.state_change_penalty,
                "eps_sim_steps": self.eps_sim_steps_train,
                "e_r_b": self.e_r_b_train,
                "g_e": self.g_e_train,
                "rew_l_b": np.min(self.e_r_b_train[1, 0, :]),
                "rew_u_b": np.max(self.e_r_b_train[1, 0, :]),
            })

        # Set type-specific parameters
        if type == "train":
            env_kwargs.update({
                "eps_ind": self.eps_ind,
                "state_change_penalty": self.EnvConfig.state_change_penalty,
                "eps_sim_steps": self.eps_sim_steps_train,
                "e_r_b": self.e_r_b_train,
                "g_e": self.g_e_train,
                "rew_l_b": np.min(self.e_r_b_train[1, 0, :]),
                "rew_u_b": np.max(self.e_r_b_train[1, 0, :]),
            })
        else:
            env_kwargs.update({
                "eps_ind": None,
                "state_change_penalty": 0.0,
            })
            if type == "val":
                env_kwargs.update({
                    "eps_sim_steps": self.eps_sim_steps_val,
                    "e_r_b": self.e_r_b_val,
                    "g_e": self.g_e_val,
                    "rew_l_b": np.min(self.e_r_b_train[1, 0, :]),
                    "rew_u_b": np.max(self.e_r_b_train[1, 0, :]),
                })
            elif type == "test":
                env_kwargs.update({
                    "eps_sim_steps": self.eps_sim_steps_test,
                    "e_r_b": self.e_r_b_test,
                    "g_e": self.g_e_test,
                    "rew_l_b": np.min(self.e_r_b_train[1, 0, :]),
                    "rew_u_b": np.max(self.e_r_b_train[1, 0, :]),
                })
            else:
                raise ValueError(f'Invalid type: {type}. Must be "train", "val", or "test".')

        return env_kwargs
                   

def initial_print():
    print('\n---------------------------------------------------------------------------------------------')    
    print('--MCTS_Q: Monte Carlo Tree Search and Deep Q Network for Power-to-Gas dispatch optimization--')
    print('---------------------------------------------------------------------------------------------\n')


def config_print(EnvConfig, TrainConfig, MCTSQConfig):
    """
        Gathers and prints general settings
        :param EnvConfig: Environment configuration (class object)
        :param TrainConfig: Training configuration (class object)
        :param MCTSQConfig: MCTS_Q configuration (class object)
        :return str_id: String for identification of the present training run
    """
    print("Set training case...")
    print(f"---Training case details: MCTS_Q_{TrainConfig.str_inv} ")
    str_id = "MCTS_Q_" + TrainConfig.str_inv
    if EnvConfig.scenario == 1: print("    > Business case (_BS):\t\t\t 1 - Trading at the electricity, gas, and emission spot markets")
    elif EnvConfig.scenario == 2: print("    > Business case  (_BS):\t\t\t 2 - Fixed synthetic natural gas (SNG) price and trading at the electricity and emission spot markets")
    else: print("    > Business case (_BS):\t\t\t 3 - Participating in EEG tenders by using a CHP plant and trading at the electricity spot markets")
    str_id += "_BS" + str(EnvConfig.scenario)
    print(f"    > Operational load level (_OP) :\t\t {EnvConfig.operation}")
    str_id += "_" + str(EnvConfig.operation)
    print(f"    > Training episode length (_ep) :\t\t {EnvConfig.eps_len_d} days")
    str_id += "_ep" + str(EnvConfig.eps_len_d)
    print(f"    > Time step size (action frequency) (_ts) :\t {EnvConfig.sim_step} seconds")
    str_id += "_ts" + str(EnvConfig.sim_step)
    print(f"    > Random seed (_rs) :\t\t\t {TrainConfig.seed_train}")
    str_id += MCTSQConfig.get_hyper()
    str_id += "_rs" + str(TrainConfig.seed_train)                       # Random seed at the end of "str_id" for file simplified file

    return str_id

class CallbackVal():
    def __init__(self, env_id, env_kwargs_data_val, val_steps):
        self.env = gym.make(env_id, dict_input=env_kwargs_data_val, train_or_eval = "eval")
        self.val_steps = val_steps  
        self.stats = {'steps': [],
                      'cum_rew': []}
        self.model = None


def create_envs(env_id, env_kwargs_data, TrainConfig):
    """Creates environments for training, validation, and testing"""

    env_train = gym.make(env_id, dict_input=env_kwargs_data['env_kwargs_train'], train_or_eval="train")
    env_test_post = gym.make(env_id, dict_input=env_kwargs_data['env_kwargs_test'], train_or_eval="eval")

    callback_val = CallbackVal(env_id, env_kwargs_data['env_kwargs_val'], val_steps=TrainConfig.val_steps)

    return env_train, callback_val, env_test_post
       
class Postprocessing():
    """A class for post-processing"""

    def __init__(self, str_id, EnvConfig, TrainConfig, env_test_post, Preprocess, model):
        """
            Initializes variables
            :param str_id: Unique identifier for the current training run.
            :param AgentConfig: Configuration settings for the agent (class object).
            :param EnvConfig: Configuration settings for the environment (class object).
            :param TrainConfig: Configuration settings for training (class object).
            :param env_test_post: Test environment instance used for post-processing.
            :param Preprocess: Instance of the Preprocessing class.
        """
        self.EnvConfig = EnvConfig
        self.TrainConfig = TrainConfig
        self.eps_sim_steps_test = Preprocess.eps_sim_steps_test
        self.env_test_post = env_test_post
        self.stats_dict_test = {}
        self.str_id = str_id
        self.model = model
        model.load(TrainConfig.log_path + str_id)

    def test_performance(self):
        """
            Test RL policy on the test environment
        """
        stats = np.zeros((self.eps_sim_steps_test, len(self.EnvConfig.stats_names)))

        obs = self.env_test_post.reset()
        timesteps = self.eps_sim_steps_test#  - 6

        for i in tqdm(range(timesteps), desc='---Apply RL policy on the test environment:'):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _ , terminated, info = self.env_test_post.step(action)

            # Store data in stats
            if not terminated:
                j = 0
                for val in info[0]:
                    if j < 24:
                        if val == 'Meth_Action':
                            if info[0][val] == 'standby':
                                stats[i, j] = 0
                            elif info[0][val] == 'cooldown':
                                stats[i, j] = 1
                            elif info[0][val] == 'startup':
                                stats[i, j] = 2
                            elif info[0][val] == 'partial_load':
                                stats[i, j] = 3
                            else:
                                stats[i, j] = 4
                        else:
                            stats[i, j] = info[0][val]
                    j += 1
        
        
        for m in range(len(self.EnvConfig.stats_names)):
            self.stats_dict_test[self.EnvConfig.stats_names[m]] = stats[:(timesteps), m]

        return None

    def plot_results(self):
        """Generates a multi-subplot plot displaying time-series data and methanation operations based on the agent's actions"""
        print("---Plot and save RL performance on the test set under ./plots/ ...\n") 

        stats_dict = self.stats_dict_test
        time_sim = stats_dict['steps_stats'] * self.EnvConfig.sim_step
        time_sim *= 1 / 3600 / 24   # Converts the simulation time into days
        time_sim = time_sim[:-6]    # ptg_gym_env.py curtails an episode by 6 time steps to ensure a data overhead
        meth_state = stats_dict['Meth_State_stats'][:-6]+1

        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True, sharey=False)
        axs[0].plot(time_sim, stats_dict['el_price_stats'][:-6], label='El.')
        axs[0].plot(time_sim, stats_dict['gas_price_stats'][:-6], 'g', label='(S)NG')
        axs[0].set_ylabel('El. & (S)NG prices\n [ct/kWh]')
        axs[0].legend(loc="upper left", fontsize='small')
        axs0_1 = axs[0].twinx()
        axs0_1.plot(time_sim, stats_dict['eua_price_stats'][:-6], 'k', label='EUA')
        axs0_1.set_ylabel('EUA prices [€/t$_{CO2}$]')
        axs0_1.legend(loc="upper right", fontsize='small')

        axs[1].plot(time_sim, meth_state, 'b', label='state')
        axs[1].set_yticks([1,2,3,4,5])
        axs[1].set_yticklabels(['Standby', 'Cooldown/Off', 'Startup', 'Partial Load', 'Full Load'])
        axs[1].set_ylabel(' ')
        axs[1].legend(loc="upper left", fontsize='small')
        axs[1].grid(axis='y', linestyle='dashed')
        axs1_1 = axs[1].twinx()
        axs1_1.plot(time_sim, stats_dict['Meth_CH4_flow_stats'][:-6]*1000, 'yellowgreen', label='CH$_4$')
        axs1_1.set_ylabel('CH$_4$ flow rate\n [mmol/s]')
        axs1_1.legend(loc="upper right", fontsize='small')  
        
        axs[2].plot(time_sim, stats_dict['Meth_reward_stats'][:-6]/100, 'g', label='Reward')
        axs[2].set_ylabel('Reward [€]')
        axs[2].set_xlabel('Time [d]')
        axs[2].legend(loc="upper left", fontsize='small')
        axs2_1 = axs[2].twinx()
        axs2_1.plot(time_sim, stats_dict['Meth_cum_reward_stats'][:-6]/100, 'k', label='Cum. Reward')
        axs2_1.set_ylabel('Cumulative \n reward [€]')
        axs2_1.legend(loc="upper right", fontsize='small')

        fig.suptitle(f"{self.str_id} \n Rew: {np.round(stats_dict['Meth_cum_reward_stats'][-7]/100, 0)} €", fontsize=9)
        plt.savefig(f'plots/{self.str_id}_plot.png')

        plt.close()
