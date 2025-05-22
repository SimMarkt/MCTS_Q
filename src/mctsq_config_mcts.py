# ----------------------------------------------------------------------------------------------------------------
# MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
# GitHub Repository: https://github.com/SimMarkt/MCTS_Q
#
# mctsq_config_mcts: 
# > Incorporates the MCTS algorithm with DQN guidance on tree search
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
import gc
import multiprocessing
from torch.utils.tensorboard import SummaryWriter

from src.mctsq_config_dqn import DQNModel

#TODO: Include random seeds
#TODO: Include deterministic=True für validation and testing??

class MCTSQConfiguration():
    """
    Configuration class for MCTS_Q algorithm.
    This class loads the configuration from a YAML file and initializes the MCTS_Q algorithm with the specified parameters.
    """

    def __init__(self):
        # Load the algorithm configuration from the YAML file
        with open("config/config_mctsq.yaml", "r") as env_file:
            mctsq_config = yaml.safe_load(env_file)

        # Unpack data from dictionary
        self.__dict__.update(mctsq_config)

        self.rl_alg_hyp = mctsq_config

        self.str_alg = ""          # Initialize the string for the algorithm settings (used for file identification)
        # Nested dictionary with hyperparameters, including abbreviation ('abb') and variable name ('var') 
        # 'var' must match the notation in MCTS_Q/config/config_mctsq.yaml
        self.hyper = {'Iterations': {'abb' :"_it", 'var': 'iterations'},
                      'PUCT Initial Exploration': {'abb' :"_pu", 'var': 'c_init'},
                      'PUCT Exploration Increase': {'abb' :"_pi", 'var': 'c_base'},
                      'Max Tree depth': {'abb' :"_md", 'var': 'maximum_depth'},
                      'Learning rate': {'abb' :"_al", 'var': 'learning_rate'},
                      'Discount factor': {'abb' :"_ga", 'var': 'discount_factor'},
                      'Replay buffer size': {'abb' :"_rb", 'var': 'buffer_size'},
                      'Batch size': {'abb' :"_bs", 'var': 'batch_size'},
                      'Hidden layers': {'abb' :"_hl", 'var': 'hidden_layers'},
                      'Hidden units': {'abb' :"_hu", 'var': 'hidden_units'},
                      'Activation function': {'abb' :"_af", 'var': 'activation'},
                      'Process data sequence': {'abb' :"_sq", 'var': 'seq_length'},
                      'Embedding dimensions': {'abb' :"_ed", 'var': 'embed_dim'},
                      'Price encoder type': {'abb' :"_ee", 'var': 'price_encoder_type'},
                      'Process encoder type': {'abb' :"_pe", 'var': 'process_encoder_type'},
                      'Gas EUA encoder type': {'abb' :"_ge", 'var': 'gas_eua_encoder_type'},
                      }         

    def get_hyper(self):
        """
            Displays the algorithm's hyperparameters and returns a string identifier for file identification.
            :return str_alg: The hyperparameter settings as a string for file identification
        """

        # Display the chosen algorithm and its hyperparameters
        print(f"    > MCTS_Q algorithm : >>> <<<")
        self.hyp_print('Iterations')
        self.hyp_print('PUCT Initial Exploration')
        self.hyp_print('PUCT Exploration Increase')
        self.hyp_print('Max Tree depth')
        self.hyp_print('Learning rate')
        self.hyp_print('Discount factor')
        self.hyp_print('Replay buffer size')
        self.hyp_print('Batch size')
        self.hyp_print('Hidden layers')
        self.hyp_print('Hidden units')
        self.hyp_print('Activation function')
        self.hyp_print('Process data sequence')
        self.hyp_print('Embedding dimensions')
        self.hyp_print('Price encoder type')
        self.hyp_print('Process encoder type')
        self.hyp_print('Gas EUA encoder type')
        print(' ')

        return self.str_alg

    def hyp_print(self, hyp_name: str):
        """
            Displays the value of a specific hyperparameter and adds it to the string identifier for file naming
            :param hyp_name: Name of the hyperparameter to display
        """
        assert hyp_name in self.hyper, f"Specified hyperparameter ({hyp_name}) is not part of the implemented settings!"
        length_str = len(hyp_name)
        if length_str > 32:         print(f"         {hyp_name} ({self.hyper[hyp_name]['abb']}): {self.rl_alg_hyp[self.hyper[hyp_name]['var']]}")
        elif length_str > 22:       print(f"         {hyp_name} ({self.hyper[hyp_name]['abb']}):\t {self.rl_alg_hyp[self.hyper[hyp_name]['var']]}")
        elif length_str > 15:       print(f"         {hyp_name} ({self.hyper[hyp_name]['abb']}):\t\t {self.rl_alg_hyp[self.hyper[hyp_name]['var']]}")
        else:                       print(f"         {hyp_name} ({self.hyper[hyp_name]['abb']}):\t\t\t {self.rl_alg_hyp[self.hyper[hyp_name]['var']]}")
        self.str_alg += self.hyper[hyp_name]['abb'] + str(self.rl_alg_hyp[self.hyper[hyp_name]['var']])


class MCTS_Q:
    def __init__(self, env_train, seed, config=None, tb_log=None):
        """
        Initialize MCTS_Q with the training environment and DQN agent.
        :param env_train: The training environment
        :param dqn: The DQN model for support of MCTS action selection
        :param seed: Random seed for reproducibility
        :param config: MCTSQConfig instance
        """
        if config is not None:
            self.__dict__.update(config.__dict__)
        else:
            # Fallback: load from YAML if config not provided
            with open("config/config_mctsq.yaml", "r") as env_file:
                mctsq_config = yaml.safe_load(env_file)
            self.__dict__.update(mctsq_config)

        self.env_train = env_train

        self.tb_log = tb_log
        self.writer = None
        if self.tb_log is not None:
            self.writer = SummaryWriter(log_dir=self.tb_log)

        # Load the environment configuration from the YAML file
        with open("config/config_env.yaml", "r") as env_file:
            env_config = yaml.safe_load(env_file)

        el_input_dim = 1                        # "Elec_Price" in ptg_gym_env observation space
        process_input_dim = 6                   # "T_CAT", "H2_in_MolarFlow", "CH4_syn_MolarFlow", "H2_res_MolarFlow", "H2O_DE_MassFlow", "Elec_Heating"
        gas_eua_input_dim = 2                   # "Gas_Price", "EUA_Price"
        temporal_encodings = 2                  # Temporal encoding with two additional features (sin/cos)

        # Initialize the DQN model
        action_dim = env_train.action_space.n

        self.dqn = DQNModel(
            el_input_dim=el_input_dim+temporal_encodings, 
            process_input_dim=process_input_dim+temporal_encodings, 
            gas_eua_input_dim=gas_eua_input_dim+temporal_encodings, 
            action_dim=action_dim, 
            embed_dim=self.embed_dim,
            hidden_layers=self.hidden_layers,
            hidden_units=self.hidden_units,
            buffer_capacity=self.buffer_size,
            batch_size=self.batch_size,
            gamma=self.discount_factor,
            lr=self.learning_rate,
            price_encoder_type=self.price_encoder_type,
            process_encoder_type=self.process_encoder_type,
            gas_eua_encoder_type=self.gas_eua_encoder_type,
            activation=self.activation,
            learning_starts=self.learning_starts,
            seed=seed
        )

        self.action_type = "discrete"

        self.deterministic = False 

    def learn(self, total_timesteps, callback):
        """
        Learn MCTS_Q parameters.
        :param total_timesteps: Total number of timesteps for training
        :param callback: Callback function for evaluation
        """

        state, _ = self.env_train.reset()      

        self.eval_processes = []

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

            # # Evaluate the policy (in parallel, non-blocking)
            # if callback is not None:
            #     if self.step % callback.val_steps == 0 and self.step != 0:
            #         callback.model = self   # Set the MCTS_Q model for the callback
            #         result_queue = multiprocessing.Queue()
            #         p = multiprocessing.Process(
            #             target=run_callback_eval,
            #             args=(self.step, callback, result_queue)
            #         )
            #         p.start()
            #         self.eval_processes.append((p, result_queue))

            # # Check for finished evaluation processes and collect results
            # for proc, queue in self.eval_processes[:]:
            #     if not proc.is_alive():
            #         proc.join()
            #         if not queue.empty():
            #             step_eval, cum_reward_call = queue.get()
            #             callback.stats.append(step_eval, cum_reward_call)
            #             print(f"   >>Cumulative Reward {cum_reward_call}")
            #         self.eval_processes.remove((proc, queue))

            if self.step % self.target_update == 0 and self.step != 0:
                self.dqn.update_target_network()
            
            # Evaluate the policy (In Serial processing)
            if callback is not None:
                if self.step % callback.val_steps == 0 and self.step != 0:
                    state_call, _ = callback.env.reset()
                    cum_reward_call = 0 

                    for _ in tqdm(range(callback.env.eps_sim_steps), desc='   >>Validation:'):
                        action_call = self.predict(callback.env)
                        _, reward_call, terminated_call, _, _ = callback.env.step(action_call)
                        cum_reward_call += reward_call
                        if terminated_call:
                            break

                    # callback.stats['steps'].append(self.step)
                    # callback.stats['cum_rew'].append(cum_reward_call)
                    print(f"   >>Cumulative Reward {cum_reward_call}")

                    if self.writer is not None:
                        self.writer.add_scalar("Validation/CumulativeReward", cum_reward_call, global_step=self.step)

        if self.writer is not None:
            self.writer.close()

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

        best_action = root_node.most_visited_child().action

        # Explicitly delete the tree to break reference cycles and clear memory
        del root_node
        gc.collect()

        return best_action
    
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
                # Use the DQN model's Q-values for the prior policy

                price_state, process_state, gas_eua_state = self._get_state(node.env)  # Get the state representation

                with torch.no_grad():
                    q_values = self.dqn.policy_net(price_state, process_state, gas_eua_state)
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
    
    def _evaluate(self, node):
        """
        Use the DQN model's value function to estimate the value of a leaf node.
        :param node: The leaf node to evaluate
        :return: The expected value of Q-values for the node
        """
        price_state, process_state, gas_eua_state = self._get_state(node.env)  # Get the state representation

        with torch.no_grad():
            q_values = self.dqn.policy_net(price_state, process_state, gas_eua_state)  # Get Q-values from the DQN
            policy_probs = F.softmax(q_values, dim=-1)  # Compute policy probabilities using softmax
            expected_value = torch.sum(policy_probs * q_values, dim=-1).item()  # Compute the expected value of Q-values
        return expected_value
        #return q_values.max().item()  # Use the maximum Q-value as the value estimate
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

    def _get_state(self, env):
        """
        Get the state representation from the environment.
        :param env: The environment to get the state from
        :return: The state representation for electricity prices, process data, and gas/EUA prices
        """

        state = env.get_obs()

        price_state = np.array(state["Elec_Price"])[..., np.newaxis]  # shape: (#el_data, 1)
        process_state = np.stack([
            state["T_CAT"],
            state["H2_in_MolarFlow"],
            state["CH4_syn_MolarFlow"],
            state["H2_res_MolarFlow"],
            state["H2O_DE_MassFlow"],
            state["Elec_Heating"]
        ], axis=-1)  # shape: (seq_length, 6)
        gas_eua_state = np.stack([
            state["Gas_Price"],
            state["EUA_Price"]
        ], axis=-1) # shape: (#gaseua_data, 1)

        price_state = torch.FloatTensor(price_state).unsqueeze(0)      # (1, sequence, 1)
        process_state = torch.FloatTensor(process_state).unsqueeze(0)  # (1, sequence, num_features)
        gas_eua_state = torch.FloatTensor(gas_eua_state).unsqueeze(0)  # (1, sequence, num_features)

        return price_state, process_state, gas_eua_state

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
    

def run_callback_eval(step, callback, result_queue):
    state_call, _ = callback.env.reset()
    cum_reward_call = 0
    for _ in tqdm(range(callback.env.eps_sim_steps), desc='   >>Validation:'): # range(callback.env.eps_sim_steps): 
        action_call = callback.model.predict(callback.env)
        _, reward_call, terminated_call, _, _ = callback.env.step(action_call)
        cum_reward_call += reward_call
        if terminated_call:
            break
    # Send results back to main process
    result_queue.put((step, cum_reward_call))
     
