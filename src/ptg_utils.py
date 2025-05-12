# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas Dispatch Optimization
# GitHub Repository: https://github.com/SimMarkt/RL_PtG
#
# rl_utils: 
# > Utiliy/Helper functions
# ----------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

from src.ptg_opt import calculate_optimum

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


def load_data(EnvConfig):
    """
        Loads historical market data and experimental methanation operation data
        :return dict_price_data: Dictionary containing electricity, gas, and EUA market data
                dict_op_data: Dictionary containing time-series data for dynamic methanation operations
    """
    print("---Load data")

    # Define market data file types (electricity, gas, and EUA) and dataset splits.
    market_data_files = [('el_price', 'el'), ('gas_price', 'gas'), ('eua_price', 'eua')]
    splits = ['test']

    dict_price_data = {
        f'{data[0]}_{split}': import_market_data(getattr(EnvConfig, f'datafile_path_{split}_{data[1]}'), data[1], EnvConfig.path)
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
        key: import_data(getattr(EnvConfig, f'datafile_path{num}'), EnvConfig.path)
        for key, num in op_data_files
    }

    if EnvConfig.scenario in [2, 3]:  
        gas_price = EnvConfig.ch4_price_fix if EnvConfig.scenario == 2 else 0
        dict_price_data['gas_price_test'] = np.full(len(dict_price_data['gas_price_test']), gas_price)

        if EnvConfig.scenario == 3:  
            dict_price_data['eua_price_test'] = np.zeros(len(dict_price_data['eua_price_test']))

    return dict_price_data, dict_op_data

class Preprocessing():
    """A class for preprocessing energy market and process data"""

    def __init__(self, dict_price_data, dict_op_data, EnvConfig):
        """
            Initializes preprocessing with configuration parameters and data
            :param dict_price_data: Dictionary containing historical market data.
            :param dict_op_data: Dictionary containing dynamic process data.
            :param AgentConfig: Configuration settings for the agent.
            :param EnvConfig: Configuration settings for the environment.
            :param TrainConfig: Configuration settings for training.
        """
        # Store configuration objects and input data
        self.EnvConfig = EnvConfig
        self.dict_price_data = dict_price_data
        self.dict_op_data = dict_op_data
        # Placeholder variables for computed values.
        self.dict_pot_r_b = None                    # Dictionary containing the potential reward [pot_rew...] and the boolean load identifier [part_full_b...]

        # e_r_b_test: (hourly values)
        #   np.array which stores elec. price data, potential reward, and boolean identifier
        #   Dimensions = [Type of data] x [No. of day-ahead values] x [historical values]
        #       Type of data = [el_price, pot_rew, part_full_b]
        #       No. of day-ahead values = EnvConfig.price_ahead
        #       historical values = No. of values in the electricity price data set
        self.e_r_b_test = None, None, None

        # g_e_test: (daily values)
        #   np.array which stores gas and EUA price data
        #   Dimensions = [Type of data] x [No. of day-ahead values] x [historical values]
        #       Type of data = [gas_price, pot_rew, part_full_b]
        #       No. of day-ahead values = 2 (today and tomorrow)
        #       historical values = No. of values in the gas/EUA price data set
        self.g_e_test = None, None, None

        # Variables for splitting the training set into randomly selected subsets for episodic learning
        self.eps_sim_steps_test = None          # No. of steps in the test set

        # Additional variables for the training processin RL_PtG (Not used in MCTS_PtG, but required for ptg_gym_env.py)
        self.n_eps = None                       # Episode length in seconds
        self.r_level = None                     # Reward level for the PtG process (0: partial load, 1: full load)  
        self.eps_ind = None                     # Indexes of the randomly ordered training subsets (Only in RL_PtG)
        self.eps_ind = None                     # Indexes of the randomly ordered training subsets (Only in RL_PtG)
        self.overhead_factor = 10               # Overhead factor for self.eps_ind to account for randomness in process selection during multiprocessing (Must be an integer)
        assert isinstance(self.overhead_factor, int), f"Episode overhead ({self.overhead_factor}) must be an integer!"
        self.n_eps_loops = None                 # Total No. of episodes across the entire training process 

        self.preprocessing_rew()                # Compute potential reward and load identifier
        self.preprocessing_array()              # Convert day-ahead datasets into NumPy arrays for calculations.
        self.define_episodes()                  # Define the number of episodes and the number of steps in each episode. (Not used in MCTS_PtG, but required for ptg_gym_env.py)      
        

    def preprocessing_rew(self):
        """
            Data preprocessing, including the calculation of potential rewards.
            These represent the maximum possible reward in Power-to-Gas (PtG) operation, 
            either in partial load [part_full_b... = 0] or full load [part_full_b... = 1].
        """

        # Compute PtG operation data for the theoretical optimum (T-OPT), ignoring system dynamics.
        # The function calculate_optimum() was moved from Preprocessing() to the separate rl_opt.py file for better clarity.
        print("---Calculate the theoretical optimum, the potential reward, and the load identifier")
        stats_dict_opt_test = calculate_optimum(self.dict_price_data['el_price_test'], self.dict_price_data['gas_price_test'],
                                                self.dict_price_data['eua_price_test'], "Test", self.EnvConfig.stats_names)

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
            'pot_rew_test': stats_dict_opt_test['Meth_reward_stats'],
            'part_full_b_test': stats_dict_opt_test['part_full_stats'],
        }


    def preprocessing_array(self):
        """Convert dictionaries to NumPy arrays for computational efficiency"""

        # e_r_b: Multi-Dimensional array which stores Day-ahead electricity price data as well as Day-ahead potential rewards
        # and load identifiers for the entire test sets
        # e.g. e_r_b_test[0, 5, 156] represents the future value of the electricity price [0,-,-] in 4 hours [-,5,-] at the
        # 156ths entry of the electricity price data set 
        self.e_r_b_test = np.zeros((3, self.EnvConfig.price_ahead, self.dict_price_data['el_price_test'].shape[0] - self.EnvConfig.price_ahead))

        for i in range(self.EnvConfig.price_ahead):    
            self.e_r_b_test[0, i, :] = self.dict_price_data['el_price_test'][i:(-self.EnvConfig.price_ahead + i)]
            self.e_r_b_test[1, i, :] = self.dict_pot_r_b['pot_rew_test'][i:(-self.EnvConfig.price_ahead + i)]
            self.e_r_b_test[2, i, :] = self.dict_pot_r_b['part_full_b_test'][i:(-self.EnvConfig.price_ahead + i)]

        # g_e: Multi-Dimensional Array which stores Day-ahead gas and EUA price data for the entire test set        
        self.g_e_test = np.zeros((2, 2, self.dict_price_data['gas_price_test'].shape[0]-1))

        self.g_e_test[0, 0, :] = self.dict_price_data['gas_price_test'][:-1]
        self.g_e_test[1, 0, :] = self.dict_price_data['eua_price_test'][:-1]
        self.g_e_test[0, 1, :] = self.dict_price_data['gas_price_test'][1:]
        self.g_e_test[1, 1, :] = self.dict_price_data['eua_price_test'][1:]

    def define_episodes(self):
        """Defines settings for training and evaluation episodes"""

        print("---Define episodes and step size limits")
        # No. of days in the test set ("-1" accounts for the day-ahead overhead
        test_len_d = len(self.dict_price_data['gas_price_test']) - 1

        # No. of steps per episode in testing
        self.eps_sim_steps_test = int(24 * 3600 * test_len_d /self.EnvConfig.sim_step)
        
        # Print training process details
        print("    > Steps in the test set =", self.eps_sim_steps_test, "\n")


    def dict_env_kwargs(self):
        """
            Returns global model parameters and hyperparameters for the PtG environment as a dictionary.
            :return: env_kwargs - Dictionary containing global parameters and hyperparameters.
        """

        # General environment configurations (Contains some variables not used in MCTS_PtG)
        env_kwargs = {
            **{f"ptg_{key}": self.EnvConfig.ptg_state_space[key] for key in 
            ["standby", "cooldown", "startup", "partial_load", "full_load"]},
            **{key: getattr(self.EnvConfig, key) for key in 
            ["noise", "eps_len_d", "sim_step", "time_step_op", "price_ahead", "scenario",
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
                "h2o_l_b", "h2o_u_b", "heat_l_b", "heat_u_b", "raw_modified"]},
            "parallel": None,
            "n_eps_loops": None,
            "reward_level": [0],
            "action_type": 'discrete',
        }

        # Operation data
        env_kwargs.update({key: self.dict_op_data[key] for key in self.dict_op_data})

        env_kwargs.update({
            "eps_sim_steps": self.eps_sim_steps_test,
            "e_r_b": self.e_r_b_test,
            "g_e": self.g_e_test,
            "rew_l_b": np.min(self.e_r_b_test[1, 0, :]),
            "rew_u_b": np.max(self.e_r_b_test[1, 0, :]),
            "eps_ind": None,
            "raw_modified": 'raw',
            "state_change_penalty": 0.0,
        })

        return env_kwargs
                   
def initial_print():
    print('\n--------------------------------------------------------------------------------------------')    
    print('----------MCTS_PtG: Monte-Carlo Tree Search for Power-to-Gas dispatch optimization----------')
    print('--------------------------------------------------------------------------------------------\n')


def config_print(mcts, EnvConfig):
    """
        Gathers and prints general settings
        :param mcts: MCTS (class object)
        :param EnvConfig: Environment configuration (class object)
        :return str_id: String for identification of the present training run
    """
    print("Set test case...")
    str_id = "MCTS_PtG_"
    if EnvConfig.scenario == 1: print("    > Business case (_BS):\t\t\t 1 - Trading at the electricity, gas, and emission spot markets")
    elif EnvConfig.scenario == 2: print("    > Business case  (_BS):\t\t\t 2 - Fixed synthetic natural gas (SNG) price and trading at the electricity and emission spot markets")
    else: print("    > Business case (_BS):\t\t\t 3 - Participating in EEG tenders by using a CHP plant and trading at the electricity spot markets")
    str_id += "_BS" + str(EnvConfig.scenario)
    print(f"    > Operational load level (_OP) :\t\t {EnvConfig.operation}")
    str_id += "_" + str(EnvConfig.operation)
    print(f"    > Time step size (action frequency) (_ts) :\t {EnvConfig.sim_step} seconds\n")
    str_id += "_ts" + str(EnvConfig.sim_step)
    print("MCTS settings...")
    print(f"    > No. of Iterations (_it) :\t\t\t {mcts.iterations}")
    str_id += "_it" + str(mcts.iterations)
    print(f"    > Exploration parameter (_ex) :\t\t {mcts.exploration_weight}")
    str_id += "_ex" + str(mcts.exploration_weight)
    print(f"    > Simulation steps (_si) :\t\t\t {mcts.total_steps}")
    str_id += "_si" + str(mcts.total_steps)
    print(f"    > Maximum depth (_md) :\t\t\t {mcts.maximum_depth}\n")
    str_id += "_md" + str(mcts.maximum_depth)
    
    return str_id 

def plot_results(stats_dict_test, EnvConfig, str_id):
        """Generates a multi-subplot plot displaying time-series data and methanation operations based on the agent's actions"""
        print("---Plot and save RL performance on the test set under ./plots/ ...\n") 

        stats_dict = stats_dict_test
        time_sim = stats_dict['steps_stats'] * EnvConfig.sim_step
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

        fig.suptitle(f"{str_id} \n Rew: {np.round(stats_dict['Meth_cum_reward_stats'][-7]/100, 0)} €", fontsize=9)
        plt.savefig(f'plots/{str_id}_plot.png')

        plt.close()
