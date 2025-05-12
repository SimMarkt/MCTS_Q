# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas Dispatch Optimization
# GitHub Repository: https://github.com/SimMarkt/RL_PtG
#
# rl_main:
# > Main script for training deep reinforcement learning (RL) algorithms on the PtG-CH4 dispatch task.
# > Adapts to different computational environments: a local personal computer ('pc') or a computing cluster with SLURM management ('slurm').
# ----------------------------------------------------------------------------------------------------------------

# --------------------------------------------Import Python libraries---------------------------------------------
import os
import torch as th

# Library for the RL environment
from gymnasium.envs.registration import registry, register
import gymnasium as gym 

# Libraries with utility functions and classes
from src.rl_utils import load_data, initial_print, config_print, Preprocessing, Postprocessing, create_vec_envs#, create_vec_envs
from src.rl_config_agent import AgentConfiguration
from src.rl_config_env import EnvConfiguration
from src.rl_config_train import TrainConfiguration

from src.mcts_q_config_dqn import DQN, DQNAgent
from src.mcts_q_config_mcts import MCTS

def computational_resources(TrainConfig):
    """
        Configures computational resources and sets the random seed for the current thread
        :param TrainConfig: Training configuration (class object)
    """
    print("Set computational resources...")
    TrainConfig.path = os.path.dirname(__file__)
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
            from env.ptg_gym_env import PTGEnv

            # Register the environment
            register(
                id=env_id,
                entry_point="env.ptg_gym_env:PTGEnv",  # Path to the environment class
            )
            print(f"---Environment '{env_id}' registered successfully!\n")
        except ImportError as e:
            print(f"Error importing the environment module: {e}")
        except Exception as e:
            print(f"Error registering the environment: {e}")
    else:
        print(f"---Environment '{env_id}' is already registered.\n")

def main():
    # --------------------------------------Initialize the RL configuration---------------------------------------
    initial_print()
    AgentConfig = AgentConfiguration()
    EnvConfig = EnvConfiguration()
    TrainConfig = TrainConfiguration()
    computational_resources(TrainConfig)
    str_id = config_print(AgentConfig, EnvConfig, TrainConfig)
    
    # -----------------------------------------------Preprocessing------------------------------------------------
    print("Preprocessing...")
    dict_price_data, dict_op_data = load_data(EnvConfig, TrainConfig)

    # Initialize preprocessing with calculation of potential rewards and load identifiers
    Preprocess = Preprocessing(dict_price_data, dict_op_data, AgentConfig, EnvConfig, TrainConfig)    
    # Create dictionaries for kwargs of training and test environments
    env_kwargs_data = {'env_kwargs_train': Preprocess.dict_env_kwargs("train"),
                       'env_kwargs_val': Preprocess.dict_env_kwargs("val"),
                       'env_kwargs_test': Preprocess.dict_env_kwargs("test"),}

    # Instantiate the vectorized environments
    print("Load environment...")
    env_id = 'PtGEnv-v0'
    check_env(env_id)
    dict_input, train_or_eval = "train", render_mode="None"
    env_train = gym.make(env_id, env_kwargs_data['env_kwargs_train'], train_or_eval = "train")  # Create the training environment
    env_eval = gym.make(env_id, env_kwargs_data['env_kwargs_val'], train_or_eval = "eval")  # Create the training environment
    env_test_post = gym.make(env_id, env_kwargs_data['env_kwargs_test'], train_or_eval = "test")

    tb_log = "tensorboard/" + str_id 
    
    # ----------------------------------------------Initialize RL Model--------------------------------------------
    print("Initializing RL model...")
    state_dim = env_train.observation_space.shape[0]
    action_dim = env_train.action_space.n
    embed_dim = 64
    hidden_units = 128

    rl_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        seq_length=10,  # Example sequence length
        embed_dim=embed_dim,
        hidden_units=hidden_units,
        buffer_capacity=10000,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        encoder_type="conv"
    )                                                                                   # Set path for tensorboard data (for monitoring RL training) 

    # ----------------------------------------------MCTS with RL--------------------------------------------------
    print("Run MCTS with RL on the validation set... >>>", str_id, "<<< \n")
    mcts = MCTS()
    mcts.path = os.path.dirname(__file__)

    mcts.run_with_rl(env_train, env_test_post, EnvConfig, Preprocess.eps_sim_steps_test, rl_agent)  # Pass RL agent to MCTS
    print("...finished MCTS validation\n")

    # model.learn(total_timesteps=TrainConfig.train_steps, callback=[eval_callback_val])  # Evaluate the RL agent only on the validation set

    # # ------------------------------------------------Save model--------------------------------------------------
    # if TrainConfig.model_conf == "save_model" or TrainConfig.model_conf == "save_load_model":
    #     print("Save RL agent under ./logs/ ... \n") 
    #     AgentConfig.save_model(model)
    
    # # ----------------------------------------------Post-processing-----------------------------------------------
    # print("Postprocessing...")
    # PostProcess = Postprocessing(str_id, AgentConfig, EnvConfig, TrainConfig, env_test_post, Preprocess)
    # PostProcess.test_performance()
    # PostProcess.plot_results()

if __name__ == '__main__':
    main()



