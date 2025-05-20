# ----------------------------------------------------------------------------------------------------------------
# RL_PtG: Power-to-Gas Dispatch Optimization using Monte Carlo Tree Search (MCTS)
# GitHub Repository: https://github.com/SimMarkt/MCTS_PtG
#
# ptg_config_mcts: 
# > Contains the source code for the MCTS algorithm with a node class and mcts class
# > Converts the data from 'config_mcts.yaml' into a class object for further processing and usage
# ----------------------------------------------------------------------------------------------------------------

import math
import random
import copy
import yaml
import csv
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F


from src.mctsq_config_dqn import DQNModel

#TODO: Include random seeds
#TODO: Include deterministic=True für validation and testing??
#TODO: Write stats during training and evaluation into tensorboard file
#TODO: Include auto exploration adjustment

class MCTS_Q:
    def __init__(self, env_train, seed):
        """
        Initialize MCTS_Q with the training environment and DQN agent.
        :param env_train: The training environment
        :param dqn: The DQN model for support of MCTS action selection
        :param seed: Random seed for reproducibility
        """
        # Load the algorithm configuration from the YAML file
        with open("config/config_mctsq.yaml", "r") as env_file:
            mctsq_config = yaml.safe_load(env_file)

        # Unpack data from dictionary
        self.__dict__.update(mctsq_config)

        self.env_train = env_train

        # Load the environment configuration from the YAML file
        with open("config/config_env.yaml", "r") as env_file:
            env_config = yaml.safe_load(env_file)

        el_input_dim = env_config["price_ahead"] + env_config["price_past"]   # Input dimension for the electricity price
        gas_eua_input_dim = 3

        # Initialize the DQN model
        action_dim = env_train.action_space.n

        self.dqn = DQNModel(
            el_input_dim=el_input_dim, 
            process_input_dim=self.seq_length, 
            gas_eua_input_dim=gas_eua_input_dim, 
            action_dim=action_dim, 
            embed_dim=self.embed_dim,
            hidden_layers=self.hidden_layers,
            hidden_units=self.hidden_units,
            buffer_capacity=self.buffer_size,
            batch_size=self.batch_size,
            gamma=self.discount_factor,
            lr=self.learning_rate,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay=self.epsilon_decay,
            price_encoder_type=self.price_encoder_type,
            process_encoder_type=self.process_encoder_type,
            gas_eua_encoder_type=self.gas_eua_encoder_type,
            activation=self.activation,
            seed=seed
        )

        self.action_type = "discrete"

        self.deterministic = False 

        self.str_alg = None          # Initialize the string for the algorithm settings (used for file identification)
        # Nested dictionary with hyperparameters, including abbreviation ('abb') and variable name ('var') 
        # 'var' must match the notation in MCTS_Q/config/config_mctsq.yaml
        self.hyper = {'Iterations': {'abb' :"_it", 'var': 'iterations'},
                      'PUCT Initial Exploration': {'abb' :"_al", 'var': 'c_init'},
                      'PUCT Exploration Increase': {'abb' :"_al", 'var': 'c_base'},
                      'Max Tree depth': {'abb' :"_al", 'var': 'maximum_depth'},
                      'Learning rate': {'abb' :"_al", 'var': 'learning_rate'},
                      'Discount factor': {'abb' :"_ga", 'var': 'discount_factor'},
                      'Initial exploration coefficient': {'abb' :"_ie", 'var': 'epsilon_start'},
                      'Final exploration coefficient': {'abb' :"_fe", 'var': 'epsilon_end'},
                      'Decay rate for exploration': {'abb' :"_re", 'var': 'epsilon_decay'},
                      'Replay buffer size': {'abb' :"_rb", 'var': 'buffer_size'},
                      'Batch size': {'abb' :"_bs", 'var': 'batch_size'},
                      'Hidden units': {'abb' :"_hu", 'var': 'hidden_units'},
                      'Encoder type': {'abb' :"_en", 'var': 'encoder_type'},
                      'Encoder sequence length': {'abb' :"_sl", 'var': 'seq_length'},
                      'Embedding dimensions': {'abb' :"_ed", 'var': 'embed_dim'},
                      } 
        

    def learn(self, total_timesteps, callback):
        """
        Learn MCTS_Q parameters.
        :param total_timesteps: Total number of timesteps for training
        :param callback: Callback function for evaluation
        """

        state, _ = self.env_train.reset()      

        for self.step in tqdm(range(total_timesteps), desc='---Training MCTS_Q:'):
            
            # Perform step based on MCTS with DQN values
            action = self.predict(self.env_train)
            next_state, reward, terminated, _, _ = self.env_train.step(action)

            # Train DQN parameter
            self.dqn.replay_buffer.push(state, action, reward, next_state, terminated)
            self.dqn.update()

            state = next_state
            if terminated:
                break
            
            # Evaluate the policy
            if callback is not None:
                if self.step % callback.val_steps == 0 and self.step != 0:
                    state_call, _ = callback.env.reset()
                    cum_reward_call = 0 

                    for _ in range(callback.env.ep_length):
                        action_call = self.predict(callback.env)
                        _, reward_call, terminated_call, _, _ = callback.env.step(action_call)
                        cum_reward_call += reward_call
                        if terminated_call:
                            break

                    callback.stats.append(self.step, cum_reward_call)
                    print(f" Validation: Cumulative Reward {cum_reward_call}")
            

    def predict(self, env, deterministic=False):
        """
        Perform MCTS search to find the best action
        :param env: The environment to search in
        """
        self.deterministic = deterministic

        root_env_copy = copy.deepcopy(env)
        root_node = MCTSNode(root_env_copy, maximum_depth=self.maximum_depth)

        for _ in range(self.iterations):
            node = self._select(root_node)
            if not node.is_terminal():
                node = self._expand(node)
            value = self._evaluate(node.env)
            self._backpropagate(node, value)

        return root_node.most_visited_child().action
    
    def _select(self, node):
        """
        Select the best child node based on PUCT (Upper Confidence Bound for Trees).
        :param node: The current node in the MCTS tree
        :return: The selected child node
        """
        while not node.is_terminal() and node.is_fully_expanded():
            # Compute PUCT scores for all children
            total_visits = sum(child.visits for child in node.children)
            c_base = self.c_base  # Hyperparameter for exploration adjustment
            c_init = self.c_init  # Hyperparameter for exploration adjustment

            # Compute exploration adjustment C(s)
            c_puct = math.log((1 + total_visits + c_base) / c_base) + c_init

            for child in node.children:
                # Use the RL model's Q-values for the prior policy
                state_tensor = torch.FloatTensor(node.env.state).unsqueeze(0)  # Add batch dimension
                with torch.no_grad():
                    q_values = self.dqn.policy_net(state_tensor)
                prior_prob = F.softmax(q_values, dim=-1)[0, child.action].item()

                # Compute the mean Q-value for the child node
                mean_q_value = child.total_value / child.visits if child.visits > 0 else 0

                # Compute PUCT score
                child.puct_score = (
                    mean_q_value + c_puct * prior_prob * math.sqrt(total_visits) / (1 + child.visits)
                    #TODO: Include auto exploration adjustment
                )

            # Select the child with the highest PUCT score
            node = max(node.children, key=lambda child: child.puct_score)

        return node
    
    def _expand(self, node):
        """
        Expand the node by adding a new child node.
        :param node: The current node in the MCTS tree
        :return: The newly created child node
        """
        if node.depth >= self.maximum_depth:  # Prevent expansion beyond max depth
            return node

        legal_actions = node.get_legal_actions()
        tried_actions = [child.action for child in node.children]
        untried_actions = [a for a in legal_actions if a not in tried_actions]

        # Select a random untried action
        action = random.choice(untried_actions)
        new_env = copy.deepcopy(node.env)
        _, _, terminated, truncated, _ = new_env.step(action)
        done = terminated or truncated
        child_node = MCTSNode(
            new_env, parent=node, action=action, done=done, maximum_depth=self.maximum_depth
        )
        node.children.append(child_node)
        return child_node
    
    # def _evaluate(self, node):
    #     """
    #     Use the RL model's value function to estimate the value of a leaf node.
    #     :param node: The leaf node to evaluate
    #     :return: The estimated value of the node
    #     """
    #     state_tensor = torch.FloatTensor(node.env.state).unsqueeze(0)  # Add batch dimension
    #     with torch.no_grad():
    #         q_values = self.dqn.policy_net(state_tensor)
    #     return q_values.max().item()  # Use the maximum Q-value as the value estimate

    def _evaluate(self, node):
        """
        Use the RL model's value function to estimate the value of a leaf node.
        :param node: The leaf node to evaluate
        :return: The expected value of Q-values for the node
        """
        state_tensor = torch.FloatTensor(node.env.state).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            q_values = self.dqn.policy_net(state_tensor)  # Get Q-values from the DQN
            policy_probs = F.softmax(q_values, dim=-1)  # Compute policy probabilities using softmax
            expected_value = torch.sum(policy_probs * q_values, dim=-1).item()  # Compute the expected value of Q-values
        return expected_value
        #TODO: Incorporate n-step return for value estimation

    def _backpropagate(self, node, value):
        """
        Backpropagate the value from the leaf node to the root node
        :param node: The current node in the MCTS tree
        :param value: The value received from the DQN model
        """
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    def save(self, filepath):
        """
        Save the DQN model parameters used by MCTS_Q.
        """
        self.dqn.save(filepath)

    def load(self, filepath):
        """
        Load the DQN model parameters into MCTS_Q.
        """
        self.dqn.load(filepath)


class MCTSNode:
    def __init__(self, env, parent=None, action=None, done=False, remaining_steps=42, total_steps=42, depth=0, maximum_depth=42):
        self.env = env
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_value = 0.0
        self.done = done                               
        if remaining_steps < total_steps:
            self.remaining_steps = remaining_steps      # No. of steps remaining for the simulation
        else:
            self.remaining_steps = total_steps
        self.depth = depth
        self.maximum_depth = maximum_depth

    def is_terminal(self):
        """
        Check if the node is terminal (i.e., if the environment is done or maximum depth is reached)
        :return: True if the node is terminal, False otherwise
        """
        return self.done or self.depth >= self.maximum_depth

    def is_fully_expanded(self):
        """
        Check if all possible actions have been tried at this node
        :return: True if all actions have been tried, False otherwise
        """
        legal_actions = self.get_legal_actions()
        tried_actions = [child.action for child in self.children]
        return set(tried_actions) == set(legal_actions)
    
    def get_legal_actions(self):
        """
        Get the legal actions for the current node based on the environment's action space
        :return: List of legal actions
        """
        meth_state = self.env.Meth_State  # Access the Meth_State from the environment
        if meth_state == 0:     # 'standby'
            return [0, 1, 2]    # Allows only standby, cooldown, and startup actions
        elif meth_state == 1:   # 'cooldown'
            return [0, 1, 2]    # Allows only standby, cooldown, and startup actions
        elif meth_state == 2:   # 'startup'
            return [0, 1, 3, 4] # Allows only standby, cooldown, and load level after startup (partial load, full load)
        elif meth_state == 3:   # 'partial load'
            return [0, 1, 3, 4] # Allows only standby, cooldown, and load level (partial load, full load)
        elif meth_state == 4:   # 'full load'
            return [0, 1, 3, 4] # Allows only standby, cooldown, and load level (partial load, full load)
        else:
            return list(range(self.env.action_space.n))  # Default to all actions
        
    def most_visited_child(self):
        """
        Select the child node with the most visits
        :return: The child node with the highest number of visits
        """
        return max(self.children, key=lambda child: child.visits)
     
