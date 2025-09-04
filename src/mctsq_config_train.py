"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

mctsq_config_train: 
> Manages the configuration and settings for MCTS_Q training.
----------------------------------------------------------------------------------------------------
"""

import yaml

class TrainConfiguration:
    def __init__(self):
        # Load the environment configuration from the YAML file
        with open("config/config_train.yaml", "r") as env_file:
            train_config = yaml.safe_load(env_file)

        # Unpack data from dictionary
        self.__dict__.update(train_config)
        
        # Initialize key attributes
        self.path = None                        # MCTS_Q folder path
        self.slurm_id = None                    # SLURM ID of a specific thread
        com_set = ['pc', 'slurm']
        
        assert self.com_conf in com_set, f"Invalid computation setup specified - data/config_train.yaml -> com_conf : {self.com_conf} must match {com_set}"
        
        assert len(self.r_seed_train) == len(self.r_seed_test), 'Training and test sets must have the same number of random seeds!'
        self.seed_train = None              # Random seed for training
        self.seed_test = None               # Random seed for validation/testing

        # Load the algorithm configuration from the YAML file
        with open("config/config_mctsq.yaml", "r") as env_file:
            mctsq_config = yaml.safe_load(env_file)

        self.seq_length = mctsq_config["seq_length"]
        self.seq_step = mctsq_config["seq_step"]

        # Load the environment configuration from the YAML file
        with open("config/config_env.yaml", "r") as env_file:
            env_config = yaml.safe_load(env_file)
        
        assert (self.seq_length * self.seq_step * env_config["time_step_op"]) >= env_config["sim_step"], f'Sequence {self.seq_length * self.seq_step * env_config["time_step_op"]} must be equal or smaller than one simulation step {env_config["sim_step"]} in the environment!'

        







       



