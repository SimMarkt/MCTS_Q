# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Deep Reinforcement Learning for Power-to-Gas Dispatch Optimization
# GitHub Repository: https://github.com/SimMarkt/RL_PtG
#
# rl_config_agent:
# > Stores hyperparameters for the RL agents
# > Parses the config_agent.yaml data into a class object for further processing
# > Includes three additional functions to specify, load, and save Stable-Baselines3 models
# -------------------------------------------------------------------------------------------

import numpy as np
import torch as th
import yaml

class AgentConfiguration:
    def __init__(self):
        # Load the environment configuration from the YAML file
        with open("config/config_agent.yaml", "r") as env_file:
            agent_config = yaml.safe_load(env_file)

        # Unpack data from dictionary
        self.__dict__.update(agent_config)

        # Ensure the specified RL algorithm exists in the hyperparameters
        assert self.rl_alg in self.hyperparameters, f"Invalid algorithm specified - data/config_agent.yaml -> model_conf : {self.rl_alg} must match {self.hyperparameters.keys()}"
        self.rl_alg_hyp = self.hyperparameters[self.rl_alg]      # Hyperparameters of the selected algorithm
        self.str_alg = None                                      # Initialize the string for the algorithm settings (used for file identification)
        # Nested dictionary with hyperparameters, including abbreviation ('abb') and variable name ('var') 
        # 'var' must match the notation in RL_PtG/config/config_agent.yaml
        self.hyper = {'Learning rate': {'abb' :"_al", 'var': 'alpha'},
                      'Discount factor': {'abb' :"_ga", 'var': 'gamma'},
                      'Initial exploration coefficient': {'abb' :"_ie", 'var': 'eps_init'},
                      'Final exploration coefficient': {'abb' :"_fe", 'var': 'eps_fin'},
                      'Exploration ratio': {'abb' :"_re", 'var': 'eps_fra'},
                      'Entropy coefficient': {'abb' :"_ec", 'var': 'ent_coeff'},
                      'Exploration noise': {'abb' :"_en", 'var': 'sigma_exp'},
                      'n-step TD update': {'abb' :"_ns", 'var': 'n_steps'},
                      'n-step factor': {'abb' :"_nf", 'var': 'n_steps_f'},
                      'Replay buffer size': {'abb' :"_rb", 'var': 'buffer_size'},
                      'Batch size': {'abb' :"_bs", 'var': 'batch_size'},
                      'Hidden layers': {'abb' :"_hl", 'var': 'hidden_layers'},
                      'Hidden units': {'abb' :"_hu", 'var': 'hidden_units'},
                      'Activation function': {'abb' :"_ac", 'var': 'activation'},
                      'Generalized advantage estimation': {'abb' :"_ge", 'var': 'gae_lambda'},
                      'No. of epochs': {'abb' :"_ep", 'var': 'n_epoch'},
                      'Normalize advantage': {'abb' :"_na", 'var': 'normalize_advantage'},
                      'No. of quantiles': {'abb' :"_nq", 'var': 'n_quantiles'},
                      'Dropped quantiles': {'abb' :"_dq", 'var': 'top_quantiles_drop'},
                      'No. of critics': {'abb' :"_cr", 'var': 'n_critics'},
                      'Soft update': {'abb' :"_ta", 'var': 'tau'},
                      'Learning starts': {'abb' :"_ls", 'var': 'learning_starts'},
                      'Training frequency': {'abb' :"_tf", 'var': 'train_freq'},
                      'Target update interval': {'abb' :"_tu", 'var': 'target_update_interval'},
                      'gSDE exploration': {'abb' :"_gs", 'var': 'gSDE'},
                      }

    def set_model(self, env, tb_log, TrainConfig):
        """
            Specifies and initializes the Stable-Baselines3 model for RL training
            :param env: The environment for training
            :param tb_log: The TensorBoard log file location
            :param TrainConfig: Training configuration (class object)
            :return model: Stable-Baselines3 model for RL training
        """

        # Implement the neural network architecture:
        #   Custom actor (pi) and value function (vf) networks with the
        #   same architecture net_arch and activation function (th.nn.ReLU, th.nn.Tanh)
        #   Note: An extra linear layer will be added on top of the pi and vf nets
        if self.rl_alg_hyp['activation'] == 'ReLU':
            activation_fn = th.nn.ReLU
        elif self.rl_alg_hyp['activation'] == 'Tanh':
            activation_fn = th.nn.Tanh
        else:
            assert False, f"Type of activation function ({self.rl_alg_hyp['activation']}) needs to be 'ReLU' or 'Tanh'! -> Check RL_PtG/config/config_agent.yaml"
        net_arch = np.ones((self.rl_alg_hyp['hidden_layers'],), int) * self.rl_alg_hyp['hidden_units']
        net_arch = net_arch.tolist()

        # Set RL algorithms and configure hyperparameters
        if self.rl_alg == 'DQN':
            from stable_baselines3 import DQN           
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch)
            model = DQN(
                "MultiInputPolicy",                                                     # Policy type
                env,                                                                    # Environment
                verbose=0,                                                              # Suppress verbose output
                tensorboard_log=tb_log,                                                 # Tensorboard log file
                learning_rate=self.rl_alg_hyp['alpha'],                                 # Learning rate
                gamma=self.rl_alg_hyp['gamma'],                                         # Discount factor
                buffer_size=int(self.rl_alg_hyp['buffer_size']),                        # Replay buffer size
                batch_size=int(self.rl_alg_hyp['batch_size']),                          # Batch size
                exploration_initial_eps=self.rl_alg_hyp['eps_init'],                    # Initial exploration coefficient
                exploration_final_eps=self.rl_alg_hyp['eps_fin'],                       # Final exploration coefficient
                exploration_fraction=self.rl_alg_hyp['eps_fra'],                        # Fraction of the training set used for exploration annealing
                learning_starts=int(self.rl_alg_hyp['learning_starts']),                # No. of timesteps before training starts
                tau=self.rl_alg_hyp['tau'],                                             # Soft update parameter
                train_freq=self.rl_alg_hyp['train_freq'],                               # Training frequency
                policy_kwargs=policy_kwargs,                                            # Network settings
                device=TrainConfig.device,                                              # Device (CPU or GPU)
                target_update_interval=self.rl_alg_hyp['target_update_interval'],       # Target network update interval
                seed=TrainConfig.seed_train,                                            # Random seed
            )

        elif self.rl_alg == 'A2C':                      
            from stable_baselines3 import A2C
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch)
            assert self.rl_alg_hyp['normalize_advantage'] in [False,True], f"Normalize advantage ({self.rl_alg_hyp['normalize_advantage']}) should be 'False' or 'True'! - Check RL_PtG/config/config_agent.yaml"
            assert self.rl_alg_hyp['gSDE'] in [False,True], f"gSDE exploration ({self.rl_alg_hyp['gSDE']}) should be 'False' or 'True'! - Check RL_PtG/config/config_agent.yaml"
            model = A2C(
                "MultiInputPolicy",                                                     # Policy type
                env,                                                                    # Environment
                verbose=0,                                                              # Suppress verbose output
                tensorboard_log=tb_log,                                                 # Tensorboard log file
                learning_rate=self.rl_alg_hyp['alpha'],                                 # Learning rate
                gamma=self.rl_alg_hyp['gamma'],                                         # Discount factor
                n_steps=self.rl_alg_hyp['n_steps'],                                     # No. of steps of the n-step TD update
                gae_lambda=self.rl_alg_hyp['gae_lambda'],                               # Factor for generalized advantage estimation
                normalize_advantage=self.rl_alg_hyp['normalize_advantage'],             # Normalize advantage
                ent_coef=self.rl_alg_hyp['ent_coeff'],                                  # Entropy coefficient
                policy_kwargs=policy_kwargs,                                            # Network settings
                use_sde=self.rl_alg_hyp['gSDE'],                                        # gSDE exploration
                device=TrainConfig.device,                                              # Device (CPU or GPU)
                seed=TrainConfig.seed_train,                                            # Random seed
            )

        elif self.rl_alg == 'PPO':                      
            from stable_baselines3 import PPO
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch)
            n_steps = int(self.rl_alg_hyp['n_steps_f'] * self.rl_alg_hyp['batch_size'])       # No. of steps of the n-step TD update
            assert self.rl_alg_hyp['normalize_advantage'] in [False,True], f"Normalize advantage ({self.rl_alg_hyp['normalize_advantage']}) should be 'False' or 'True'! - Check RL_PtG/config/config_agent.yaml"
            assert self.rl_alg_hyp['gSDE'] in [False,True], f"gSDE exploration ({self.rl_alg_hyp['gSDE']}) should be 'False' or 'True'! - Check RL_PtG/config/config_agent.yaml"
            model = PPO(
                "MultiInputPolicy",                                                     # Policy type
                env,                                                                    # Environment
                verbose=0,                                                              # Suppress verbose output
                tensorboard_log=tb_log,                                                 # Tensorboard log file
                learning_rate=self.rl_alg_hyp['alpha'],                                 # Learning rate
                gamma=self.rl_alg_hyp['gamma'],                                         # Discount factor
                batch_size=int(self.rl_alg_hyp['batch_size']),                          # Batch size
                n_steps=n_steps,                                                        # No. of steps of the n-step TD update
                gae_lambda=self.rl_alg_hyp['gae_lambda'],                               # Factor for generalized advantage estimation
                n_epochs=int(self.rl_alg_hyp['n_epoch']),                               # No. of epochs for mini batch training
                normalize_advantage=self.rl_alg_hyp['normalize_advantage'],             # Normalize advantage
                ent_coef=self.rl_alg_hyp['ent_coeff'],                                  # Entropy coefficient
                policy_kwargs=policy_kwargs,                                            # Network settings
                use_sde=self.rl_alg_hyp['gSDE'],                                        # gSDE exploration
                device=TrainConfig.device,                                              # Device (CPU or GPU)
                seed=TrainConfig.seed_train,                                            # random seed
            )

        elif self.rl_alg == 'TD3':                    
            from stable_baselines3 import TD3
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch)
            model = TD3(
                "MultiInputPolicy",                                                     # Policy type
                env,                                                                    # Environment
                verbose=0,                                                              # Suppress verbose output
                tensorboard_log=tb_log,                                                 # Tensorboard log file
                learning_rate=self.rl_alg_hyp['alpha'],                                 # Learning rate
                gamma=self.rl_alg_hyp['gamma'],                                         # Discount factor
                buffer_size=int(self.rl_alg_hyp['buffer_size']),                        # Replay buffer size
                batch_size=int(self.rl_alg_hyp['batch_size']),                          # Batch size
                learning_starts=int(self.rl_alg_hyp['learning_starts']),                # No. of timesteps before training starts
                tau=self.rl_alg_hyp['tau'],                                             # Soft update parameter
                train_freq=self.rl_alg_hyp['train_freq'],                               # Training frequency
                target_policy_noise=self.rl_alg_hyp['sigma_exp'],                       # Standard deviation of Gaussian noise added to the target policy
                policy_kwargs=policy_kwargs,                                            # Network settings
                device=TrainConfig.device,                                              # Device (CPU or GPU)
                seed=TrainConfig.seed_train,                                            # Random seed
            )
        
        elif self.rl_alg == 'SAC':                    
            from stable_baselines3 import SAC
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch)
            assert self.rl_alg_hyp['gSDE'] in [False,True], f"gSDE exploration ({self.rl_alg_hyp['gSDE']}) should be 'False' or 'True'! - Check RL_PtG/config/config_agent.yaml"
            model = SAC(
                "MultiInputPolicy",                                                     # Policy type
                env,                                                                    # Environment
                verbose=0,                                                              # Suppress verbose output
                tensorboard_log=tb_log,                                                 # Tensorboard log file
                learning_rate=self.rl_alg_hyp['alpha'],                                 # Learning rate
                gamma=self.rl_alg_hyp['gamma'],                                         # Discount factor
                buffer_size=int(self.rl_alg_hyp['buffer_size']),                        # Replay buffer size
                batch_size=int(self.rl_alg_hyp['batch_size']),                          # Batch size
                learning_starts=int(self.rl_alg_hyp['learning_starts']),                # No. of timesteps before training starts
                tau=self.rl_alg_hyp['tau'],                                             # Soft update parameter
                train_freq=self.rl_alg_hyp['train_freq'],                               # Training frequency
                ent_coef=self.rl_alg_hyp['ent_coeff'],                                  # Entropy coefficient
                policy_kwargs=policy_kwargs,                                            # Network settings
                use_sde=self.rl_alg_hyp['gSDE'],                                        # gSDE exploration
                device=TrainConfig.device,                                              # Device (CPU or GPU)
                target_update_interval=self.rl_alg_hyp['train_freq'],                   # Target network update interval
                seed=TrainConfig.seed_train,                                            # Random seed
            )
        
        elif self.rl_alg == 'TQC':             
            from sb3_contrib import TQC
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch,
                            n_critics=int(self.rl_alg_hyp['n_critics']), n_quantiles=int(self.rl_alg_hyp['n_quantiles']))
            assert self.rl_alg_hyp['gSDE'] in [False,True], f"gSDE exploration ({self.rl_alg_hyp['gSDE']}) should be 'False' or 'True'! - Check RL_PtG/config/config_agent.yaml"
            model = TQC(
                "MultiInputPolicy",                                                         # Policy type
                env,                                                                        # Environment
                verbose=0,                                                                  # Suppress verbose output
                tensorboard_log=tb_log,                                                     # Tensorboard log file
                top_quantiles_to_drop_per_net=int(self.rl_alg_hyp['top_quantiles_drop']),   # No. of top quantiles to drop
                learning_rate=self.rl_alg_hyp['alpha'],                                     # Learning rate
                gamma=self.rl_alg_hyp['gamma'],                                             # Discount factor
                buffer_size=int(self.rl_alg_hyp['buffer_size']),                            # Replay buffer size
                batch_size=int(self.rl_alg_hyp['batch_size']),                              # Batch size
                learning_starts=int(self.rl_alg_hyp['learning_starts']),                    # No. of timesteps before training starts
                tau=self.rl_alg_hyp['tau'],                                                 # Soft update parameter
                train_freq=self.rl_alg_hyp['train_freq'],                                   # Training frequency
                ent_coef=self.rl_alg_hyp['ent_coeff'],                                      # Entropy coefficient
                policy_kwargs=policy_kwargs,                                                # Network settings
                use_sde=self.rl_alg_hyp['gSDE'],                                            # gSDE exploration
                device=TrainConfig.device,                                                  # Device (CPU or GPU)
                target_update_interval=self.rl_alg_hyp['train_freq'],                       # Target network update interval
                seed=TrainConfig.seed_train,                                                # Random seed
            )
        else:
            assert False, 'Algorithm is not implemented!'
        
        return model
    
    def load_model(self, env, tb_log, model_path, type):
        """
            Loads a pretrained Stable-Baselines3 model for RL training
            :param env: The environment for training
            :param tb_log: The file path for TensorBoard logs
            :param model_path: Path to the pretrained model
            :param type: Specifies whether to load the replay buffer ('train' for training, others for evaluation)
            :return model: Stable-Baselines3 model for RL training
        """

        # Load the pre-trained RL model based on the specified algorithm
        if self.rl_alg == 'DQN':
            from stable_baselines3 import DQN           
            model = DQN.load(model_path, tensorboard_log=tb_log)
        elif self.rl_alg == 'A2C':                     
            from stable_baselines3 import A2C
            model = A2C.load(model_path, tensorboard_log=tb_log)
        elif self.rl_alg == 'PPO':                      
            from stable_baselines3 import PPO
            model = PPO.load(model_path, tensorboard_log=tb_log)
        elif self.rl_alg == 'TD3':                     
            from stable_baselines3 import TD3
            model = TD3.load(model_path, tensorboard_log=tb_log)
        elif self.rl_alg == 'SAC':                     
            from stable_baselines3 import SAC
            model = SAC.load(model_path, tensorboard_log=tb_log)
        elif self.rl_alg == 'TQC':                      
            from sb3_contrib import TQC
            model = TQC.load(model_path, tensorboard_log=tb_log)
        else:
            assert False, 'Algorithm is not implemented!'

        if (type == 'train') and ('buffer_size' in self.rl_alg_hyp.keys()):
            model.load_replay_buffer(model_path)
            print(f"---Replay buffer loaded - size {model.replay_buffer.size()} transitions")
            model.set_env(env)

        return model

    def save_model(self, model):
        """
            Saves the trained Stable-Baselines3 model and its replay buffer (if applicable).
            :param model: Stable-Baselines3 model
        """
        model.save(self.path_files + self.str_inv)
        if 'buffer_size' in self.rl_alg_hyp.keys():
            model.save_replay_buffer(self.path_files + self.str_inv)

    def get_hyper(self):
        """
            Displays the algorithm's hyperparameters and returns a string identifier for file identification.
            :return str_alg: The hyperparameter settings as a string for file identification
        """

        # Display the chosen algorithm and its hyperparameters
        print(f"    > Deep RL agorithm : >>> {self.rl_alg} <<<")
        self.str_alg = "_" + self.rl_alg
        self.hyp_print('Learning rate')
        self.hyp_print('Discount factor')
        if self.rl_alg == 'DQN': 
            self.hyp_print('Initial exploration coefficient')
            self.hyp_print('Final exploration coefficient')
            self.hyp_print('Exploration ratio')
        if self.rl_alg in ['A2C','PPO','SAC','TQC']: self.hyp_print('Entropy coefficient')
        if self.rl_alg in ['TD3']: self.hyp_print('Exploration noise')
        if self.rl_alg in ['A2C']: self.hyp_print('n-step TD update')
        if self.rl_alg in ['PPO']:
            self.hyp_print('n-step factor')
            print(f"         No. of steps of the n-step TD update:\t {int(self.rl_alg_hyp['n_steps_f'] * self.rl_alg_hyp['batch_size'])}")
        if self.rl_alg in ['DQN','TD3','SAC','TQC']: self.hyp_print('Replay buffer size')
        if self.rl_alg in ['DQN','PPO','TD3','SAC','TQC']: self.hyp_print('Batch size')
        self.hyp_print('Hidden layers')
        self.hyp_print('Hidden units')
        self.hyp_print('Activation function')
        if self.rl_alg in ['A2C','PPO']: self.hyp_print('Generalized advantage estimation')
        if self.rl_alg == 'PPO': self.hyp_print('No. of epochs')
        if self.rl_alg in ['A2C','PPO']: self.hyp_print('Normalize advantage')
        if self.rl_alg == 'TQC':
            self.hyp_print('No. of quantiles')
            self.hyp_print('Dropped quantiles')
            self.hyp_print('No. of critics')
        if self.rl_alg in ['DQN','TD3','SAC','TQC']: 
            self.hyp_print('Soft update') 
            self.hyp_print('Learning starts') 
            self.hyp_print('Training frequency')
        if self.rl_alg == 'DQN': self.hyp_print('Target update interval')
        if self.rl_alg in ['A2C','PPO','SAC','TQC']: self.hyp_print('gSDE exploration')
        print(' ')

        return self.str_alg


    def hyp_print(self, hyp_name: str):
        """
            Displays the value of a specific hyperparameter and adds it to the string identifier for file naming
            :param hyp_name: Name of the hyperparameter to display
        """
        assert hyp_name in self.hyper, f"Specified hyperparameter ({hyp_name}) is not part of the implemented settings!"
        length_str = len(hyp_name) 
        if length_str > 28:         print(f"         {hyp_name} ({self.hyper[hyp_name]['abb']}): {self.rl_alg_hyp[self.hyper[hyp_name]['var']]}")
        elif length_str > 22:       print(f"         {hyp_name} ({self.hyper[hyp_name]['abb']}):\t {self.rl_alg_hyp[self.hyper[hyp_name]['var']]}")
        elif length_str > 15:       print(f"         {hyp_name} ({self.hyper[hyp_name]['abb']}):\t\t {self.rl_alg_hyp[self.hyper[hyp_name]['var']]}")
        else:                       print(f"         {hyp_name} ({self.hyper[hyp_name]['abb']}):\t\t\t {self.rl_alg_hyp[self.hyper[hyp_name]['var']]}")
        self.str_alg += self.hyper[hyp_name]['abb'] + str(self.rl_alg_hyp[self.hyper[hyp_name]['var']])
