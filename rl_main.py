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

# Libraries with utility functions and classes
from src.rl_utils import load_data, initial_print, config_print, Preprocessing, Postprocessing, create_vec_envs#, create_vec_envs
from src.rl_config_agent import AgentConfiguration
from src.rl_config_env import EnvConfiguration
from src.rl_config_train import TrainConfiguration

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
    check_env(env_id)                                                                                                   # Check the Gymnasium environment registry
    env_train, env_test_post, eval_callback_val, eval_callback_test = create_vec_envs(env_id, str_id, AgentConfig, TrainConfig, env_kwargs_data)          # Create vectorized environments
    tb_log = "tensorboard/" + str_id                                                                                    # Set path for tensorboard data (for monitoring RL training) 

    # Set up the RL model with the specified algorithm
    if TrainConfig.model_conf == "simple_train" or TrainConfig.model_conf == "save_model":          # Train RL model from scratch
        model = AgentConfig.set_model(env_train, tb_log, TrainConfig)
    else:                                                                                           # Load a pretrained model
        model = AgentConfig.load_model(env_train, tb_log, f"{TrainConfig.path}{TrainConfig.path_files}{str_id}", 'train')

    # ------------------------------------------------RL Training-------------------------------------------------
    print("Training... >>>", str_id, "<<< \n")
    if TrainConfig.val_n_test:  model.learn(total_timesteps=TrainConfig.train_steps, callback=[eval_callback_val, eval_callback_test])  # Evaluate the RL agent on both validation and test sets
    else:                       model.learn(total_timesteps=TrainConfig.train_steps, callback=[eval_callback_val])                      # Evaluate the RL agent only on the validation set
    print("...finished RL training\n")

    # ------------------------------------------------Save model--------------------------------------------------
    if TrainConfig.model_conf == "save_model" or TrainConfig.model_conf == "save_load_model":
        print("Save RL agent under ./logs/ ... \n") 
        AgentConfig.save_model(model)
    
    # ----------------------------------------------Post-processing-----------------------------------------------
    print("Postprocessing...")
    PostProcess = Postprocessing(str_id, AgentConfig, EnvConfig, TrainConfig, env_test_post, Preprocess)
    PostProcess.test_performance()
    PostProcess.plot_results()

if __name__ == '__main__':
    main()



