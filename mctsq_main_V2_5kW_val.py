# ----------------------------------------------------------------------------------------------------------------
# MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
# GitHub Repository: https://github.com/SimMarkt/MCTS_Q
#
# mctsq_main_V2:
# > Main script for training the MCTS_Q algorithm on the PtG-CH4 dispatch task.
# > Adapts to different computational environments: a local personal computer ('pc') or a computing cluster with SLURM management ('slurm').
# > Includes modifications to improve training speed:
#           - Incorporates ptg_gym_env_V2.py which requires the current state index (to avoid expensive deepcopy of the environment in MCTS)
#           - Accelerates inference using jit based compiling of the DQN model and batch inference
# ----------------------------------------------------------------------------------------------------------------

# --------------------------------------------Import Python libraries---------------------------------------------
import os
import torch as th

# Library for the RL environment
from gymnasium.envs.registration import registry, register

# Libraries with utility functions and classes
from src.mctsq_utils import load_data, initial_print, config_print, Preprocessing, create_envs, plot_results
from src.mctsq_config_env import EnvConfiguration
from src.mctsq_config_train import TrainConfiguration
from src.mctsq_config_mcts_V2 import MCTSQConfiguration, MCTS_Q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#TODO: FOR TRAINING, VALIDATION AND TESTING ADD PRICE_PAST NUMBER OF VALUES AT THE BEGINNING OF THE TEST SET TO ALIGN WITH FORMER TESTS
#TODO: Set up everything on HPC (Use RANGE instead of tqdm) and include real training data and test/validation data
#TODO: Try out tensorboard and save and load of the model
#TODO: Im einfachen MCTS nochmal ein Ergebnis einfügen als Abbildung

def computational_resources(TrainConfig):
    """
        Configures computational resources and sets the random seed for the current thread
        :param TrainConfig: Training configuration (class object)
    """
    print("Set computational resources...")
    TrainConfig.path = os.path.dirname(__file__)
    TrainConfig.log_path = TrainConfig.path + TrainConfig.log_path
    TrainConfig.tb_path = TrainConfig.path + TrainConfig.tb_path
    if TrainConfig.com_conf == 'pc': 
        print("---Computation on local resources")
        TrainConfig.seed_train = TrainConfig.r_seed_train[0]
        TrainConfig.seed_test = TrainConfig.r_seed_test[0]
    else: 
        print("---SLURM Task ID:", os.environ['SLURM_PROCID'])
        TrainConfig.slurm_id = int(os.environ['SLURM_PROCID'])         # Thread ID of the specific SLURM process in parallel computing on a computing cluster
        assert TrainConfig.slurm_id <= len(TrainConfig.r_seed_train), f"No. of SLURM threads exceeds the No. of specified random seeds ({len(TrainConfig.r_seed_train)}) - please add additional seed values to RL_PtG/config/config_train.yaml -> r_seed_train & r_seed_test"
        TrainConfig.seed_train = TrainConfig.r_seed_train[TrainConfig.slurm_id]
        TrainConfig.seed_test = TrainConfig.r_seed_test[TrainConfig.slurm_id]
    if TrainConfig.device == 'cpu':    print("---Utilization of CPU\n")
    elif TrainConfig.device == 'auto': print("---Automatic hardware utilization (GPU, if possible)\n")
    else:                       print("---CUDA available:", th.cuda.is_available(), "GPU device:", th.cuda.get_device_name(0), "\n")

def check_env(env_id):
    """
        Registers the Gymnasium environment if it is not already in the registry
        :param env_id: Unique identifier for the environment
    """
    if env_id not in registry:      # Check if the environment is already registered
        try:
            # Import the ptg_gym_env environment
            from env.ptg_gym_env_V2 import PTGEnv

            # Register the environment
            register(
                id=env_id,
                entry_point="env.ptg_gym_env_V2:PTGEnv",  # Path to the environment class
            )
            print(f"---Environment '{env_id}' registered successfully!\n")
        except ImportError as e:
            print(f"Error importing the environment module: {e}")
        except Exception as e:
            print(f"Error registering the environment: {e}")
    else:
        print(f"---Environment '{env_id}' is already registered.\n")

def import_data(csvfile: str):
    """
        Import different operation states of the methanation reactor
        columns: Time [min];T_cat [°C];n_h2 [mol/s];n_ch4 [mol/s];m_DE [kg/h];Pel [W];Plast [%]
    """
    # Import historic Data from csv file
    # For further information of the procedure, look at the comments in el_grid.py
    file_path = os.path.dirname(__file__) + "/" + csvfile
    df = pd.read_csv(file_path, delimiter=";", decimal=".")

    return df

def df_to_nparray(df: pd.DataFrame):
    """
        Converts the dataframe with 
        columns = ["Time [s]", "T_cat [°C]", "n_h2 [mol/s]", "n_ch4 [mol/s]", "n_h2_res [mol/s]", "m_DE [kg/h]", "Pel [W]"]
        into an np.array
    """
    t = df["Time [s]"].values.astype(float)
    t_cat = df["T_cat [gradC]"].values.astype(float)
    n_h2 = df["n_h2 [mol/s]"].values.astype(float)
    n_ch4 = df["n_ch4 [mol/s]"].values.astype(float)
    n_h2_res = df["n_h2_res [mol/s]"].values.astype(float)
    m_h2o = df["m_DE [kg/h]"].values.astype(float)
    p_el = df["Pel [W]"].values.astype(float)
    arr = np.c_[t, t_cat, n_h2, n_ch4, n_h2_res, m_h2o, p_el]

    return arr

def main():
    # -------------------------------Initialize the configurations and the model----------------------------------
    initial_print()
    EnvConfig = EnvConfiguration()
    TrainConfig = TrainConfiguration()
    MCTSQConfig = MCTSQConfiguration()
    computational_resources(TrainConfig)

    str_id = config_print(EnvConfig, TrainConfig, MCTSQConfig)
    
    # -----------------------------------------------Preprocessing------------------------------------------------
    print("Preprocessing...")
    dict_price_data, dict_op_data = load_data(EnvConfig, TrainConfig)

    # Initialize preprocessing with calculation of potential rewards and load identifiers
    Preprocess = Preprocessing(dict_price_data, dict_op_data, EnvConfig, TrainConfig)    
    # Create dictionaries for kwargs of training and test environments
    env_kwargs_data = {'env_kwargs_train': Preprocess.dict_env_kwargs("train"),
                       'env_kwargs_val': Preprocess.dict_env_kwargs("val"),
                       'env_kwargs_test': Preprocess.dict_env_kwargs("test"),}

    # Instantiate the vectorized environments
    print("Load environment...")
    env_id = 'PtGEnv-v2'
    check_env(env_id)
    env_train, callback_val, env_test_post = create_envs(env_id, env_kwargs_data, TrainConfig)

    datafile_path19 = "data_val/data-meth_validation.csv"

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
    # TOTAL_SIM_STEPS = 104833

    df_validation = import_data(datafile_path19)
    np_validation = df_to_nparray(df_validation)

    env_meth = env_test_post

    _, info = env_meth.reset()
    state_c = info['state_c']

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
    
    # action = 6
    # actions = np.array([2, 11, 10, 11, 10, 6, 2, 11, 10, 11, 10, 11, 10, 11, 9, 2, 11, 10, 11, 10, 9, 6])
    # action_steps = np.array([272, 4663, 12588, 13470, 18185, 19297, 34853, 39796, 43001, 43780, 44170, 44406, 47471, 47909,
    #                         50612, 52969, 55706, 55806, 59002, 59462, 62129, 76319])
    
    def get_env_action(action):
        if action == 9:
            return 0
        elif action == 6:
            return 1
        elif action == 2:
            return 2
        elif action == 10:
            return 3
        elif action == 11:
            return 4
        
    Meth_State_records = np.zeros((timesteps,))
    Meth_T_cat_records = np.zeros((timesteps,))
    Meth_H2_flow_records = np.zeros((timesteps,))
    Meth_H2_res_flow_records = np.zeros((timesteps,))
    Meth_CH4_flow_records = np.zeros((timesteps,))
    Meth_H2O_flow_records = np.zeros((timesteps,))
    Meth_el_heating_records = np.zeros((timesteps,))
    Meth_hot_cold_records = np.zeros((timesteps,))
    
    for i in tqdm(range(timesteps), desc="Simulation"):
        # print("Prediction Step " + str(i))
        for k in range(len(actions)):
            if (i*SIM_STEP) >= np.around(action_steps[k]):
                action = actions[k]
        if i == action_steps[0]:
            action = 2
        Meth_Actions[i] = action
        action = get_env_action(action)
        _,_,_,_,info = env_meth.step([action, state_c])
        state_c = info['state_c']
        Meth_State_records[i] = info["Meth_State"]
        Meth_T_cat_records[i] = info["Meth_T_cat"].item()
        Meth_H2_flow_records[i] = info["Meth_H2_flow"].item()
        Meth_H2_res_flow_records[i] = info["Meth_CH4_flow"].item()
        Meth_CH4_flow_records[i] = info["Meth_CH4_flow"].item()
        Meth_H2O_flow_records[i] = info["Meth_H2O_flow"].item()
        Meth_el_heating_records[i] = info["Meth_el_heating"].item()
        Meth_hot_cold_records[i] = info["Meth_Hot_Cold"]


    # ----------------------------------------------------------------------------------------------------------------------
    print("Plot results...")
    time_sim = np.zeros((timesteps,))
    time_val = np.zeros((timesteps+1,))
    for i in range(timesteps):
        time_sim[i] = i * TIME_STEP_SIZE_SIM
        time_val[i] = i * TIME_STEP_SIZE_OP
    time_val[-1] = timesteps * TIME_STEP_SIZE_OP

    print(max(Meth_CH4_flow_records))
    print(min(Meth_CH4_flow_records))

    Meth_state = Meth_State_records
    T_cat = Meth_T_cat_records
    Meth_H2_flow = Meth_H2_flow_records
    Meth_H2_res_flow = Meth_H2_res_flow_records
    Meth_CH4_flow = Meth_CH4_flow_records
    Meth_H2O_flow = Meth_H2O_flow_records
    Meth_el_heating = Meth_el_heating_records
    Meth_Hot_Cold = Meth_hot_cold_records

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

    # print("RMSE: T_cat - ", round(calculate_RMSE(T_cat+273.15, T_cat_val+273.15),1))
    # print("RMSE: Meth_H2_flow - ", round(calculate_RMSE(Meth_H2_flow, Meth_H2_flow_val)*1000,6))
    # print("RMSE: Meth_CH4_flow - ", round(calculate_RMSE(Meth_CH4_flow, Meth_CH4_flow_val)*1000,6))
    # print("RMSE: Meth_H2_res_flow - ", round(calculate_RMSE(Meth_H2_res_flow, Meth_H2_res_flow_val)*1000,6))
    # print("RMSE: Meth_H2O_flow - ", round(calculate_RMSE(Meth_H2O_flow, Meth_H2O_flow_val),6))
    # print("RMSE: Meth_el_heating - ", round(calculate_RMSE(Meth_el_heating, Meth_el_heating_val),0))

    # print("MAPE: T_cat - ", round(calculate_MAPE(T_cat+273.15, T_cat_val+273.15),1))
    # print("MAPE: Meth_H2_flow - ", round(calculate_MAPE(Meth_H2_flow, Meth_H2_flow_val),1))
    # print("MAPE: Meth_CH4_flow - ", round(calculate_MAPE(Meth_CH4_flow, Meth_CH4_flow_val),1))
    # print("MAPE: Meth_H2_res_flow - ", round(calculate_MAPE(Meth_H2_res_flow, Meth_H2_res_flow_val),1))
    # print("MAPE: Meth_H2O_flow - ", round(calculate_MAPE(Meth_H2O_flow, Meth_H2O_flow_val),1))
    # print("MAPE: Meth_el_heating - ", round(calculate_MAPE(Meth_el_heating, Meth_el_heating_val),1))



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

    # print(len(T_cat), time_sim.shape)

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


    # # -------------------------------------------Initialize MCTS_Q------------------------------------------------
    # print("Initialize MCTS_Q agent...")
    # if TrainConfig.model_conf != "test_model":
        # model = MCTS_Q(env_train, seed=TrainConfig.seed_train, config=MCTSQConfig, tb_log=TrainConfig.tb_path + str_id)
        # if TrainConfig.model_conf == "load_model" or TrainConfig.model_conf == "save_load_model":
        #     model.load(TrainConfig.log_path + str_id)       # Load pretrained model parameters


        # # -------------------------------------------Training of MCTS_Q-------------------------------------------
        # print("Training of MCTS_Q... >>>", str_id, "<<< \n")
        # model.learn(total_timesteps=TrainConfig.train_steps, callback=callback_val)

        # print("...finished training!\n")

        # # ------------------------------------------------Save model----------------------------------------------
        # if TrainConfig.model_conf == "save_model" or TrainConfig.model_conf == "save_load_model":
        #     print("Save MCTS_Q agent under ./logs/ ... \n") 
        #     model.save(TrainConfig.log_path + str_id)
    
    # # ----------------------------------------------Post-processing-----------------------------------------------
    # if TrainConfig.model_conf != "simple_train":
    #     print("Postprocessing...")
    #     # Initialize MCTS_Q for the test environment and load pretrained model parameters
    #     model_test = MCTS_Q(env_test_post, seed=TrainConfig.seed_train, config=MCTSQConfig)
    #     model_test.load(TrainConfig.log_path + str_id)
    #     stats_dict_test = model_test.test(EnvConfig, Preprocess.eps_sim_steps_test)

    #     # Plot and save results
    #     plot_results(EnvConfig=EnvConfig, stats_dict_test=stats_dict_test, str_id=str_id)

if __name__ == '__main__':
    main()



