# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas Dispatch Optimization
# GitHub Repository: https://github.com/SimMarkt/RL_PtG
#
# rl_config_train: 
# > Manages the configuration and settings for RL training.
# > Converts data from 'config_train.yaml' into a class object for further processing
# ----------------------------------------------------------------------------------------------------------------

import yaml

class TrainConfiguration:
    def __init__(self):
        # Load the environment configuration from the YAML file
        with open("config/config_train.yaml", "r") as env_file:
            train_config = yaml.safe_load(env_file)

        # Unpack data from dictionary
        self.__dict__.update(train_config)
        
        # Initialize key attributes
        self.path = None                        # RL_PtG folder path
        self.slurm_id = None                    # SLURM ID of a specific thread
        com_set = ['pc', 'slurm']
        
        assert self.com_conf in com_set, f"Invalid computation setup specified - data/config_train.yaml -> com_conf : {self.com_conf} must match {com_set}"
        train_set = ['simple_train', 'save_model', 'load_model', 'save_load_model']
        assert self.model_conf in train_set, f"Invalid training setup specified - data/config_agent.yaml -> model_conf : {train_config['model_conf']} must match {train_set}"

        assert len(self.r_seed_train) == len(self.r_seed_test), 'Training and test sets must have the same number of random seeds!'
        self.seed_train = None              # Random seed for training
        self.seed_test = None               # Random seed for validation/testing







       



