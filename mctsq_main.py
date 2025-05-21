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
from src.mctsq_utils import load_data, initial_print, config_print, Preprocessing, Postprocessing, create_envs
from src.mctsq_config_env import EnvConfiguration
from src.mctsq_config_train import TrainConfiguration
from src.mctsq_config_mcts import MCTSQConfiguration, MCTS_Q

#TODO: FOR TRAINING, VALIDATION AND TESTING ADD PRICE_PAST NUMBER OF VALUES AT THE BEGINNING OF THE TEST SET TO ALIGN WITH FORMER TESTS

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
    env_id = 'PtGEnv-v1'
    check_env(env_id)
    env_train, callback_val, env_test_post = create_envs(env_id, env_kwargs_data, TrainConfig)

    # -------------------------------------------Initialize MCTS_Q------------------------------------------------
    print("Initialize MCTS_Q agent...")
    model = MCTS_Q(env_train, seed=TrainConfig.seed_train, config=MCTSQConfig)

    # -------------------------------------------Training of MCTS_Q-----------------------------------------------
    print("Training of MCTS_Q... >>>", str_id, "<<< \n")
    model.learn(total_timesteps=TrainConfig.train_steps, callback=callback_val)

    print("...finished training!\n")

    # ------------------------------------------------Save model--------------------------------------------------
    if TrainConfig.model_conf == "save_model" or TrainConfig.model_conf == "save_load_model":
        print("Save MCTS_Q agent under ./logs/ ... \n") 
        model.save(TrainConfig.path + str_id + "/weights")
    
    # ----------------------------------------------Post-processing-----------------------------------------------
    print("Postprocessing...")
    PostProcess = Postprocessing(str_id, EnvConfig, TrainConfig, env_test_post, Preprocess, model)
    PostProcess.test_performance()
    PostProcess.plot_results()

if __name__ == '__main__':
    main()



