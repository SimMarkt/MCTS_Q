"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

mctsq_config_train: 
> Manages the configuration and settings for MCTS_Q training.
----------------------------------------------------------------------------------------------------
"""

# pylint: disable=no-member

import yaml

class TrainConfiguration:
    """ Configuration of the training procedure with MCTS_Q. """
    def __init__(self):
        # Load the environment configuration from the YAML file
        with open("config/config_train.yaml", "r", encoding="utf-8") as env_file:
            train_config = yaml.safe_load(env_file)

        # Unpack data from dictionary
        self.__dict__.update(train_config)

        # Initialize key attributes
        self.path = None                        # MCTS_Q folder path
        self.slurm_id = None                    # SLURM ID of a specific thread
        com_set = ['pc', 'slurm']

        com_set = ['pc', 'slurm']
        if self.com_conf not in com_set:
            raise ValueError(
                "Invalid computation setup specified - data/config_train.yaml -> com_conf "
                f": {self.com_conf} must match {com_set}"
            )

        assert len(self.r_seed_train) == len(self.r_seed_test), (
            'Training and test sets must have the same number of random seeds!'
        )
        self.seed_train = None              # Random seed for training
        self.seed_test = None               # Random seed for validation/testing

        # Load the algorithm configuration from the YAML file
        with open("config/config_mctsq.yaml", "r", encoding="utf-8") as env_file:
            mctsq_config = yaml.safe_load(env_file)

        self.seq_length = mctsq_config["seq_length"]
        self.seq_step = mctsq_config["seq_step"]

        # Load the environment configuration from the YAML file
        with open("config/config_env.yaml", "r", encoding="utf-8") as env_file:
            env_config = yaml.safe_load(env_file)

        seq_total = self.seq_length * self.seq_step * env_config["time_step_op"]
        assert seq_total >= env_config["sim_step"], (
            f'Sequence {self.seq_length * self.seq_step * env_config["time_step_op"]} must '
            f'be equal or smaller than one simulation step {env_config["sim_step"]} in'
            ' the environment!'
        )
