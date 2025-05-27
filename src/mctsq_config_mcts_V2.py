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

import time

from src.mctsq_config_dqn_V2 import DQNModel

#TODO: Include random seeds
#TODO: Include deterministic=True für validation and testing??
#TODO: only store best model

class MCTSQConfiguration():
    """
    Configuration class for MCTS_Q algorithm.
    This class loads the configuration from a YAML file and initializes the MCTS_Q algorithm with the specified parameters.
    """

    def __init__(self):
        # Load the algorithm configuration from the YAML file
        with open("config/config_mctsq.yaml", "r") as mctsq_file:
            mctsq_config = yaml.safe_load(mctsq_file)

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
    def __init__(self, env, seed, config=None, log_path=None, tb_log=None):
        """
        Initialize MCTS_Q with the training environment and DQN agent.
        :param env: The environment
        :param dqn: The DQN model for support of MCTS action selection
        :param seed: Random seed for reproducibility
        :param config: MCTSQConfig instance
        """
        if config is not None:
            self.__dict__.update(config.__dict__)
        else:
            # Fallback: load from YAML if config not provided
            with open("config/config_mctsq.yaml", "r") as mctsq_file:
                mctsq_config = yaml.safe_load(mctsq_file)
            self.__dict__.update(mctsq_config)

        self.env = env

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
        temporal_encodings = 0                  # (Optional) Temporal encoding with two additional features (sin/cos)

        seq_len_price = env_config['price_ahead'] + env_config['price_past']
        seq_len_gas_eua = 3
        price_step_minutes = 60
        process_step_minutes = self.seq_step * env_config['time_step_op'] / 60
        gas_eua_step_minutes = 60 * 12

        # Initialize the DQN model
        action_dim = env.action_space.n

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
            seq_len_price=seq_len_price,
            seq_len_process=self.seq_length,
            seq_len_gas_eua=seq_len_gas_eua,
            price_step_minutes=price_step_minutes,
            process_step_minutes=process_step_minutes,
            gas_eua_step_minutes=gas_eua_step_minutes,
            seed=seed
        )

        # TorchScript: Script the policy network for faster inference
        self.dqn.policy_net.eval()
        self.dqn.policy_net = torch.jit.script(self.dqn.policy_net)

        self.action_type = "discrete"

        self.deterministic = False

        self.callback_run = False

        self.root_node = None
        self.tree_remain = False

        self.train_eval = "train"  # Default mode for training

        self.time_deepcopy = 0
        self.time_step = 0
        self.time_mcts_core = 0
        self.time_select = 0
        self.time_expand = 0
        self.time_eva = 0
        self.time_back = 0
        self.time_inf = 0
        self.time_average = []
        self.time_average_copy = []

        # Initialize variables for storing tree structure (for debugging purposes)
        self.time_store = False  # Flag to store the tree structure
        self.log_path=log_path
        self.tree_log = []  # List to store tree structure
         
        self.Meth_State_tl = None
        self.el_price_tl = None
        self.pot_reward_tl = None

    def learn(self, total_timesteps, callback):
        """
        Learn MCTS_Q parameters.
        :param total_timesteps: Total number of timesteps for training
        :param callback: Callback function for evaluation
        """

        state, info = self.env.reset()  

        state_c_learn = info['state_c']

        self.callback = callback    

        # self.eval_processes = []

        for self.step in tqdm(range(total_timesteps), desc='---Training MCTS_Q:'):
            if self.step % self.store_interval == 0 and self.step != 0:
                self.time_store = True

            # start_total = time.time()
            # self.time_deepcopy = 0
            # self.time_step = 0
            # self.time_mcts_core = 0
            # self.time_select = 0
            # self.time_expand = 0
            # self.time_eva = 0
            # self.time_back = 0
            # self.time_inf = 0

            # Perform step based on MCTS with DQN values
            action = self.predict(state_c_learn, train_eval="train")
            # start_step = time.time()
            next_state, reward, terminated, _, info = self.env.step([action, state_c_learn])
            state_c_learn = info['state_c']  # Update the state for the next step
            # step_duration = time.time() - start_step
            # self.time_step += step_duration

            # Train DQN parameter
            self.dqn.replay_buffer.push(state, action, reward, next_state, terminated)
            loss = self.dqn.update()

            if (self.writer is not None) and (loss is not None):
                self.writer.add_scalar("Training/Loss", loss, global_step=self.step)

            state = next_state

            # Log the tree structure and save it to a CSV file if time_store is True
            if self.time_store:
                self._log_tree_structure(self.root_node)
                self._save_tree_to_csv("Root")
                self.time_store = False  # Reset the flag after saving
                self.tree_log = [] 
          
            # --- Keep subtree for next step ---
            if self.root_node is not None:
                # # Compute the maximum depth of the current tree
                # def get_max_depth(node):
                #     if not node.children:
                #         return node.depth
                #     return max(get_max_depth(child) for child in node.children)
                # max_depth = get_max_depth(self.root_node)

                # Find the child corresponding to the action taken
                matching_children = [child for child in self.root_node.children if child.action == action]
                if matching_children:
                    new_root = matching_children[0]
                    new_root.parent = None  # Detach from previous root
                    self.root_node = new_root
                else:
                    self.root_node = None  # No matching child, start fresh
            else:
                self.root_node = None  # No previous tree

            # # Log the tree structure and save it to a CSV file if time_store is True
            # if self.time_store:
            #     print(id(self.root_node))
            #     self._log_tree_structure(self.root_node)
            #     self._save_tree_to_csv("Child")
            #     self.time_store = False  # Reset the flag after saving
            #     self.tree_log = [] 

            #     self._log_tree_structure(new_root)
            #     self._save_tree_to_csv("Child_new")
            #     self.time_store = False  # Reset the flag after saving
            #     self.tree_log = [] 

            if terminated:
                state, info = self.env.reset()
                state_c_learn = info['state_c']  # Reset the state for the next episode

            if self.step % self.tree_remain_interval == 0 and self.step != 0:
                self.tree_remain = False

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
            if self.callback is not None:
                if self.step % self.callback.val_steps == 0 and self.step != 0:
                    self.callback_run = True

                    _, info = self.callback.env.reset()
                    state_c_call = info['state_c']  # Reset the state for the next episode
                    cum_reward_call = 0 

                    for _ in tqdm(range(self.callback.env.eps_sim_steps), desc='   >>Validation:'):
                        action_call = self.predict(state_c_call, train_eval="eval")
                        _, _, terminated_call, _, info = self.callback.env.step([action_call, state_c_call])
                        state_c_call = info['state_c']  # Update the state for the next step

                        if terminated_call:
                            break

                    # callback.stats['steps'].append(self.step)
                    # callback.stats['cum_rew'].append(cum_reward_call)
                    print(f"   >>Cumulative Reward {info['cum_reward']}")

                    if self.writer is not None:
                        self.writer.add_scalar("Validation/CumulativeReward", cum_reward_call, global_step=self.step)
                    
                    self.callback_run = False
                    self.tree_remain = True         # Create a new root_node after validation
        
            # total_duration = time.time() - start_total
            # print("[DEBUG] Time Analysis-----------")
            # print(f"     mcts core: {self.time_mcts_core/total_duration * 100}%")
            # print(f"     select: {self.time_select/total_duration * 100}%")
            # print(f"     expand core: {self.time_expand/total_duration * 100}%")
            # print(f"     eva core: {self.time_eva/total_duration * 100}%")
            # print(f"     back core: {self.time_back/total_duration * 100}%")
            # print(f"\n     deepcopy: {self.time_deepcopy/total_duration * 100}%")
            # print(f"     step: {self.time_step/total_duration * 100}%")
            # print(f"     inf: {self.time_inf/total_duration * 100}%")

            # self.time_average.append(self.time_inf)
            # print(f"     average inference time: {np.mean(self.time_average)} seconds")
            # self.time_average_copy.append(self.time_deepcopy)
            # print(f"     average deepcopy time: {np.mean(self.time_average_copy)} seconds")

            # print(f"     Replay buffer size: {len(self.dqn.replay_buffer)} samples")

            # if self.root_node is not None:
            #     print(f"     Current MCTS tree depth: {max_depth}")
            # else:
            #     print(f"     Current MCTS tree depth: 0")

            # # Log the tree structure and save it to a CSV file if time_store is True
            # if self.time_store:
            #     self._log_tree_structure(self.root_node)
            #     self._save_tree_to_csv()
            #     self.time_store = False  # Reset the flag after saving

        if self.writer is not None:
            self.writer.close()

    def predict(self, state_c, train_eval="train", deterministic=False):
        """
        Perform MCTS search to find the best action
        :param env: The environment to search in
        """
        self.deterministic = deterministic
        self.train_eval = train_eval
        # start_deepcopy = time.time()
        # root_env_copy = copy.deepcopy(env)
        # deepcopy_duration = time.time() - start_deepcopy
        # self.time_deepcopy += deepcopy_duration

        # --- Use existing subtree if available ---
        # if self.root_node is not None and self.root_node.state_c == state_c:
        if self.root_node is not None and self.tree_remain and train_eval == "train":
            root_node = self.root_node
            # Set the depth of all nodes in the subtree, starting from root_node with depth=0
            def reset_depths(node, depth=0):
                node.depth = depth
                for child in node.children:
                    reset_depths(child, depth + 1)
            reset_depths(root_node)

        else:
            root_node = MCTSNode(state_c, maximum_depth=self.maximum_depth)
            self.root_node = root_node
            self.tree_remain = True


        # root_node = MCTSNode(state_c, maximum_depth=self.maximum_depth)

        # start_mcts_core = time.time()
        for _ in range(self.iterations):
            # start_select = time.time()
            node = self._select(root_node)
            # select_duration = time.time() - start_select
            # self.time_select += select_duration
            
            if not node.is_terminal():
                # start_expand = time.time()
                node = self._expand(node)
                # expand_duration = time.time() - start_expand
                # self.time_expand += expand_duration

            # start_eva = time.time()
            value = self._evaluate(node.state_c)
            # eva_duration = time.time() - start_eva
            # self.time_eva += eva_duration

            # start_back = time.time()
            self._backpropagate(node, value)
            # back_duration = time.time() - start_back
            # self.time_back += back_duration

        # mcts_core_duration = time.time() - start_mcts_core
        # self.time_mcts_core += mcts_core_duration

        best_action = root_node.most_visited_child().action

        # # Explicitly delete the tree to break reference cycles and clear memory
        # del root_node
        # gc.collect()

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

            # # Serial inference
            # for child in node.children:
            #     # Use the DQN model's Q-values for the prior policy

            #     price_state, process_state, gas_eua_state = self._get_state(node.env)  # Get the state representation

            #     start_inf = time.time()
            #     with torch.no_grad():
            #         q_values = self.dqn.policy_net(price_state, process_state, gas_eua_state)
            #     prior_prob = F.softmax(q_values, dim=-1)[0, child.action].item()
            #     inf_duration = time.time() - start_inf
            #     self.time_inf += inf_duration

            #     # Compute the mean Q-value for the child node
            #     mean_q_value = child.total_value / child.visits if child.visits > 0 else 0

            #     # Compute PUCT score
            #     child.puct_score = (
            #         mean_q_value + c_puct * prior_prob * math.sqrt(total_visits) / (1 + child.visits)
            #     )

            # --- Batch inference for all children ---
            price_states, process_states, gas_eua_states = [], [], []
            actions = []
            for child in node.children:
                price_state, process_state, gas_eua_state = self._get_state(child.state_c)  # Get the state representation
                price_states.append(price_state)
                process_states.append(process_state)
                gas_eua_states.append(gas_eua_state)
                actions.append(child.action)

            # Concatenate tensors along batch dimension
            price_states = torch.cat(price_states, dim=0)
            process_states = torch.cat(process_states, dim=0)
            gas_eua_states = torch.cat(gas_eua_states, dim=0)

            # start_inf = time.time()
            with torch.no_grad():
                q_values_batch = self.dqn.policy_net(price_states, process_states, gas_eua_states)  # (num_children, num_actions)
                prior_probs = F.softmax(q_values_batch, dim=-1)  # (num_children, num_actions)
            # inf_duration = time.time() - start_inf
            # self.time_inf += inf_duration

            for idx, child in enumerate(node.children):
                prior_prob = prior_probs[idx, actions[idx]].item()
                child.mean_q_value = child.total_value / child.visits if child.visits > 0 else 0
                child.c_puct = c_puct
                child.prior_prob = prior_prob
                child.puct_explore = c_puct * prior_prob * math.sqrt(total_visits) / (1 + child.visits)
                child.puct_score = (
                    child.mean_q_value + child.puct_explore
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

        state_c = node.state_c

        legal_actions = node.get_legal_actions()
        tried_actions = [child.action for child in node.children]
        untried_actions = [a for a in legal_actions if a not in tried_actions]

        # Select a random untried action
        action = random.choice(untried_actions)
        # start_deepcopy = time.time()
        # new_env = copy.deepcopy(node.env)
        # deepcopy_duration = time.time() - start_deepcopy
        # self.time_deepcopy += deepcopy_duration
        # start_step = time.time()
        if self.callback_run: 
            _, _, terminated, truncated, info = self.callback.env.step([action, state_c])
            state_c = info['state_c']  # Update the state for the next step
        else: 
            next_state, reward, terminated, truncated, info = self.env.step([action, state_c])
            state_c = info['state_c']  # Update the state for the next step
            if self.train_eval=="train":
                state = state_c['obs_norm']  # Update the state for the next step

                # Train DQN parameter
                self.dqn.replay_buffer.push(state, action, reward, next_state, terminated)

        # step_duration = time.time() - start_step
        # self.time_step += step_duration
        
        done = terminated or truncated
        child_node = MCTSNode(
            state_c, parent=node, action=action, done=done, depth=node.depth+1, maximum_depth=self.maximum_depth, 
            Meth_state_tr=info['Meth_State'], el_price_tr=info['el_price_act'], pot_reward_tr=info["Pot_Reward"]
        )
        node.children.append(child_node)
        return child_node
    
    def _evaluate(self, state_c):
        """
        Use the DQN model's value function to estimate the value of a leaf node.
        :param node: The leaf node to evaluate
        :return: The expected value of Q-values for the node
        """
        price_state, process_state, gas_eua_state = self._get_state(state_c)  # Get the state representation

        with torch.no_grad():
            q_values = self.dqn.policy_net(price_state, process_state, gas_eua_state)  # Get Q-values from the DQN
            # policy_probs = F.softmax(q_values, dim=-1)  # Compute policy probabilities using softmax
            # expected_value = torch.sum(policy_probs * q_values, dim=-1).item()  # Compute the expected value of Q-values
        # return expected_value
        return q_values.max().item()  # Use the maximum Q-value as the value estimate
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

    def _get_state(self, state_c):
        """
        Get the state representation from the environment.
        :param env: The environment to get the state from
        :return: The state representation for electricity prices, process data, and gas/EUA prices
        """

        state = state_c['obs_norm']

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

    def test(self, EnvConfig, eps_sim_steps_test):
        """
            Test MCTS_Q on the test environment
        """
        _, info = self.env.reset()  
        state_c_test = info['state_c']  # Reset the state for the next episode

        stats = np.zeros((eps_sim_steps_test, len(EnvConfig.stats_names)))    
        stats_dict_test={}

        for i in tqdm(range(eps_sim_steps_test), desc='---Apply MCTS_Q in the test environment:'):
            
            # Perform step based on MCTS with DQN values
            action = self.predict(state_c_test, train_eval="eval")
            _, _, terminated, _, info = self.env.step([action, state_c_test])
            state_c_test = info['state_c']  # Update the state for the next step
            
            if terminated:
                break
            else:
                j = 0
                for val in info:
                    if j < 24:
                        if val == 'Meth_Action':
                            if info[val] == 'standby':
                                stats[i, j] = 0
                            elif info[val] == 'cooldown':
                                stats[i, j] = 1
                            elif info[val] == 'startup':
                                stats[i, j] = 2
                            elif info[val] == 'partial_load':
                                stats[i, j] = 3
                            else:
                                stats[i, j] = 4
                        else:
                            stats[i, j] = info[val]
                    j += 1

            
        for m in range(len(EnvConfig.stats_names)):
            stats_dict_test[EnvConfig.stats_names[m]] = stats[:(eps_sim_steps_test), m]

        print(f"   >>Cumulative Reward {info['cum_reward']}")
        
        return stats_dict_test
    
    def _log_tree_structure(self, node):
        """
        Recursively log the tree structure
        :param node: The current node in the MCTS tree
        """
        node_id = id(node)
        self.tree_log.append({
            "depth": node.depth,
            "node_id": node_id,
            "parent_id": id(node.parent) if node.parent else None,
            "action": node.action,
            "visits": node.visits,
            "prior_prob": node.prior_prob,
            "c_puct": node.c_puct,
            "puct_explore": node.puct_explore,
            "mean_q_value": node.mean_q_value,
            "puct_score": node.puct_score,
            "Meth_State": node.Meth_state_tr,
            "el_price": node.el_price_tr,
            "pot_reward": node.pot_reward_tr,
        })
        for child in node.children:
            self._log_tree_structure(child)

    def _save_tree_to_csv(self, stri):
        """
        Save the logged tree structure to a CSV file
        """
        with open(self.log_path + f"tree_structure_{stri}_step{self.step}.csv", "w", newline="") as csvfile:
            fieldnames = ["depth", "node_id", "parent_id", "action", "visits", "prior_prob", "c_puct", "puct_explore", "mean_q_value", "puct_score", "Meth_State", "el_price", "pot_reward"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.tree_log)


class MCTSNode:
    def __init__(self, state_c, parent=None, action=None, done=False, remaining_steps=42, total_steps=42, depth=0, maximum_depth=42, Meth_state_tr=0, el_price_tr=0, pot_reward_tr=0):
        self.state_c = state_c
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
        self.prior_prob = 0.0
        self.c_puct = 0.0
        self.puct_explore = 0.0
        self.mean_q_value = 0.0 # initialize mean  q value
        self.puct_score = 0.0  # Initialize PUCT score
        self.Meth_state_tr = Meth_state_tr
        self.el_price_tr = el_price_tr
        self.pot_reward_tr = pot_reward_tr

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
        meth_state = self.state_c['Meth_State']  # Access the Meth_State from the environment
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
            return list(range(5))  # Default to all actions
        
    def most_visited_child(self):
        """
        Select the child node with the most visits
        :return: The child node with the highest number of visits
        """
        return max(self.children, key=lambda child: child.visits)
    
# def run_callback_eval(step, callback, result_queue):
#     state_call, _, state_c_call = callback.env.reset()
#     cum_reward_call = 0
#     for _ in tqdm(range(callback.env.eps_sim_steps), desc='   >>Validation:'): # range(callback.env.eps_sim_steps): 
#         action_call = callback.model.predict(callback.env)
#         _, reward_call, terminated_call, _, _, state_c_call = callback.env.step(action_call, state_c_call)
#         cum_reward_call += reward_call
#         if terminated_call:
#             break
#     # Send results back to main process
#     result_queue.put((step, cum_reward_call))
     
