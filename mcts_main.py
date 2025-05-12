# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Power-to-Gas Dispatch Optimization using Monte Carlo Tree Search (MCTS)
# GitHub Repository: https://github.com/SimMarkt/MCTS_PtG
#
# mcts_main:
# > Main script for the PtG-CH4 dispatch optimization.
# > Adapts to different computational environments: a local personal computer ('pc') or a computing cluster with SLURM management ('slurm').
# ----------------------------------------------------------------------------------------------------------------

# --------------------------------------------Import Python libraries---------------------------------------------
import os

# Library for the RL environment
from gymnasium.envs.registration import registry, register
import gymnasium as gym

# Libraries with utility functions and classes
from src.ptg_utils import load_data, initial_print, config_print, Preprocessing, plot_results
from src.mctsq_config_mcts import MCTS
from src.ptg_config_env import EnvConfiguration

import torch
from src.mctsq_config_dqn import DQN, DQNAgent

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
    EnvConfig = EnvConfiguration()
    mcts = MCTS()
    EnvConfig.path = os.path.dirname(__file__)
    mcts.path = os.path.dirname(__file__)

    str_id = config_print(mcts, EnvConfig)
    
    # -----------------------------------------------Preprocessing------------------------------------------------
    print("Preprocessing...")
    dict_price_data, dict_op_data = load_data(EnvConfig)

    # Initialize preprocessing with calculation of potential rewards and load identifiers
    Preprocess = Preprocessing(dict_price_data, dict_op_data, EnvConfig)    
    # Create dictionaries for kwargs of training and test environments

    # Instantiate the environment
    print("Load environment...")
    env_id = 'PtGEnv-v0'
    check_env(env_id)              # Check the Gymnasium environment registry
    env_train = gym.make(env_id, dict_input = Preprocess.dict_env_kwargs("train"))  # Create the training environment
    env_test_post = gym.make(env_id, dict_input = Preprocess.dict_env_kwargs())
       
    # ----------------------------------------------MCTS Validation-----------------------------------------------
    print("Run MCTS on the validation set... >>>", str_id, "<<< \n")
    mcts.run(env_test_post, EnvConfig, Preprocess.eps_sim_steps_test)  # Run MCTS on the test environment
    print("...finished MCTS validation\n")
    
    # # ----------------------------------------------Postprocessing------------------------------------------------
    # print("Postprocessing...")
    plot_results(mcts.stats_dict_test, EnvConfig, str_id)

if __name__ == '__main__':
    main()



