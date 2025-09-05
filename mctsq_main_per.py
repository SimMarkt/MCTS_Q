"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

mctsq_main_per:
> Main script for training the MCTS_Q algorithm on the PtG-CH4 dispatch task.
> Adapts to different computational environments: 
        - a local personal computer ('pc') or 
        - a computing cluster with SLURM management ('slurm').
> Includes modifications to improve training speed:
        - Incorporates ptg_gym_env_per.py which requires the current state index 
          (to avoid expensive deepcopy of the environment in MCTS)
        - Accelerates inference using jit based compiling of the DQN model and batch inference
----------------------------------------------------------------------------------------------------
"""

# pylint: disable=no-member, import-outside-toplevel

# -------------------------------------Import Python libraries--------------------------------------
import os
import torch as th

# Library for the RL environment
from gymnasium.envs.registration import registry, register

# Libraries with utility functions and classes
from src.mctsq_utils import load_data, initial_print, config_print, create_envs, plot_results
from src.mctsq_utils import Preprocessing
from src.mctsq_config_env import EnvConfiguration
from src.mctsq_config_train import TrainConfiguration
from src.mctsq_config_mcts_per import MCTSQConfiguration, MCTS_Q

def computational_resources(train_config: TrainConfiguration) -> None:
    """
    Configures computational resources and sets the random seed for the current thread
    :param train_config: Training configuration (class object)
    """
    print("Set computational resources...")
    train_config.path = os.path.dirname(__file__)
    train_config.log_path = train_config.path + train_config.log_path
    train_config.tb_path = train_config.path + train_config.tb_path
    if train_config.com_conf == 'pc':
        print("---Computation on local resources")
        train_config.seed_train = train_config.r_seed_train[0]
        train_config.seed_test = train_config.r_seed_test[0]
    else:
        print("---SLURM Task ID:", os.environ['SLURM_PROCID'])
         # Thread ID of the specific SLURM process in parallel computing on a computing cluster
        train_config.slurm_id = int(os.environ['SLURM_PROCID'])
        assert train_config.slurm_id <= len(train_config.r_seed_train), (
            f"No. of SLURM threads exceeds the No. of specified random seeds "
            f"({len(train_config.r_seed_train)}) - please add additional seed values to "
            "MCTS_Q/config/config_train.yaml -> r_seed_train & r_seed_test"
            )
        train_config.seed_train = train_config.r_seed_train[train_config.slurm_id]
        train_config.seed_test = train_config.r_seed_test[train_config.slurm_id]
    if train_config.device == 'cpu':
        print("---Utilization of CPU\n")
    elif train_config.device == 'auto':
        print("---Automatic hardware utilization (GPU, if possible)\n")
    else:
        print("---CUDA available:", th.cuda.is_available(),
              "GPU device:", th.cuda.get_device_name(0), "\n")

def check_env(env_id: str) -> None:
    """
    Registers the Gymnasium environment if it is not already in the registry
    :param env_id: Unique identifier for the environment
    """
    if env_id not in registry:      # Check if the environment is already registered
        try:
            # Import the ptg_gym_env environment
            from env.ptg_gym_env_per import PTGEnv

            # Register the environment
            register(
                id=env_id,
                entry_point="env.ptg_gym_env_per:PTGEnv",  # Path to the environment class
            )
            print(f"---Environment '{env_id}' registered successfully!\n")
        except ImportError as e:
            print(f"Error importing the environment module: {e}")
        except Exception as e:
            print(f"Error registering the environment: {e}")
    else:
        print(f"---Environment '{env_id}' is already registered.\n")

def main() -> None:
    """
    Main function to set up and execute the RL training process.
    """
    # -----------------------------Initialize the RL configuration-------------------------------
    initial_print()
    env_config = EnvConfiguration()
    train_config = TrainConfiguration()
    mctsq_config = MCTSQConfiguration()
    computational_resources(train_config)

    str_id = config_print(env_config, train_config, mctsq_config)

    # --------------------------------------Preprocessing----------------------------------------
    print("Preprocessing...")
    dict_price_data, dict_op_data = load_data(env_config, train_config)

    # Initialize preprocessing with calculation of potential rewards and load identifiers
    preprocess = Preprocessing(dict_price_data, dict_op_data, env_config, train_config)
    # Create dictionaries for kwargs of training and test environments
    env_kwargs_data = {'env_kwargs_train': preprocess.dict_env_kwargs("train"),
                       'env_kwargs_val': preprocess.dict_env_kwargs("val"),
                       'env_kwargs_test': preprocess.dict_env_kwargs("test"),}

    # Instantiate the vectorized environments
    print("Load environment...")
    env_id = 'PtGEnv-v2'
    check_env(env_id)
    env_train, callback_val, env_test_post = create_envs(env_id, env_kwargs_data, train_config)

    # ------------------------------------Initialize MCTS_Q-----------------------------------------
    print("Initialize MCTS_Q agent...")
    if train_config.model_conf != "test_model":
        model = MCTS_Q(
            env_train,
            seed=train_config.seed_train,
            config=mctsq_config,
            log_path=train_config.log_path,
            tb_log=train_config.tb_path + str_id
        )
        if train_config.model_conf in ["load_model", "save_load_model"]:
            model.load(train_config.log_path + str_id)       # Load pretrained model parameters


        # ------------------------------------Training of MCTS_Q------------------------------------
        print("Training of MCTS_Q... >>>", str_id, "<<< \n")
        model.learn(total_timesteps=train_config.train_steps, callback=callback_val)
        print("...finished training!\n")

        # -----------------------------------------Save model---------------------------------------
        if train_config.model_conf in ["save_model", "save_load_model"]:
            print("Save MCTS_Q agent under ./logs/ ... \n")
            model.save(train_config.log_path + str_id)

    # ---------------------------------------Post-processing----------------------------------------
    if train_config.model_conf != "simple_train":
        print("Postprocessing...")
        # Initialize MCTS_Q for the test environment and load pretrained model parameters
        model_test = MCTS_Q(env_test_post, seed=train_config.seed_train, config=mctsq_config)
        model_test.load(train_config.log_path + str_id)
        stats_dict_test = model_test.test(env_config, preprocess.eps_sim_steps_test)

        # Plot and save results
        plot_results(env_config=env_config, stats_dict_test=stats_dict_test, str_id=str_id)

if __name__ == '__main__':
    main()
