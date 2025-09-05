"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

mctsq_config_mcts: 
> Incorporates the MCTS algorithm with DQN guidance on tree search
> Uses a deep copy of the environment in the MCTS procedure
----------------------------------------------------------------------------------------------------
"""

# pylint: disable=no-member

import math
import random
import copy
import gc
from typing import Any

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gymnasium as gym

from src.mctsq_config_dqn import DQNModel
from src.mctsq_utils import CallbackVal
from src.mctsq_config_env import EnvConfiguration

class MCTSQ:
    """ MCTS_Q algorithm with DQN guidance."""
    def __init__(
            self,
            env: gym.Env,
            seed: int,
            config: "MCTSQConfiguration" | None = None,
            tb_log: str | None = None
        ) -> None:
        """
        Initialize MCTS_Q with the training environment and DQN agent.
        :param env: The environment
        :param dqn: The DQN model for support of MCTS action selection
        :param seed: Random seed for reproducibility
        :param config: MCTSQConfig instance
        :param tb_log: Path for Tensorboard log
        """
        if config is not None:
            self.__dict__.update(config.__dict__)
        else:
            with open("config/config_mctsq.yaml", "r", encoding="utf-8") as env_file:
                mctsq_config = yaml.safe_load(env_file)
            self.__dict__.update(mctsq_config)

        self.env = env
        self.step = None

        self.tb_log = tb_log
        self.writer = None
        if self.tb_log is not None:
            self.writer = SummaryWriter(log_dir=self.tb_log)

        self.eval_processes = []

        # "Elec_Price" in ptg_gym_env observation space
        el_input_dim = 1
        # "T_CAT", "H2_in_MolarFlow", "CH4_syn_MolarFlow", "H2_res_MolarFlow",
        # "H2O_DE_MassFlow", "Elec_Heating"
        process_input_dim = 6
        # "Gas_Price", "EUA_Price"
        gas_eua_input_dim = 2
        # Temporal encoding with two additional features (sin/cos)
        temporal_encodings = 2

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
            seed=seed
        )

        self.action_type = "discrete"

        self.deterministic = False

    def learn(self, total_timesteps: int, callback: CallbackVal) -> None:
        """
        Learn MCTS_Q parameters.
        :param total_timesteps: Total number of timesteps for training
        :param callback: Callback function for evaluation
        """

        state, _ = self.env.reset()

        # for self.step in tqdm(range(total_timesteps), desc="Training MCTS_Q", unit="step"):
        for self.step in range(total_timesteps):
            # Perform step based on MCTS with DQN values
            action = self.predict(self.env)
            next_state, reward, terminated, _, _ = self.env.step(action)

            # Train DQN parameter
            self.dqn.replay_buffer.push(state, action, reward, next_state, terminated)
            loss = self.dqn.update()

            if (self.writer is not None) and (loss is not None):
                self.writer.add_scalar("Training/Loss", loss, global_step=self.step)

            state = next_state
            if terminated:
                break

            if self.step % self.target_update == 0 and self.step != 0:
                self.dqn.update_target_network()

            # Evaluate the policy (In Serial processing)
            if callback is not None:
                if self.step % callback.val_steps == 0 and self.step != 0:
                    _, _ = callback.env.reset()
                    cum_reward_call = 0

                    # for _ in tqdm(
                    #   range(callback.env.eps_sim_steps),
                    #   desc="Validation MCTS_Q",
                    #   unit="step"
                    # ):
                    for _ in range(callback.env.eps_sim_steps):
                        action_call = self.predict(callback.env)
                        _, _, terminated_call, _, _ = callback.env.step(action_call)

                        if terminated_call:
                            break

                    print(f"   >>Step: {self.step} - Cumulative Reward: {cum_reward_call}")

                    if self.writer is not None:
                        self.writer.add_scalar(
                            "Validation/CumulativeReward",
                            cum_reward_call,
                            global_step=self.step
                        )

        if self.writer is not None:
            self.writer.close()

    def predict(self, env: gym.Env, deterministic: bool = False) -> int:
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

    def _select(self, node: "MCTSNode") -> "MCTSNode":
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

            # Serial inference
            for child in node.children:
                # Use the DQN model's Q-values for the prior policy
                # First get the state representation
                price_state, process_state, gas_eua_state = self._get_state(child.env)

                with torch.no_grad():
                    q_values = self.dqn.policy_net(price_state, process_state, gas_eua_state)
                prior_prob = F.softmax(q_values, dim=-1)[0, child.action].item()

                # Compute the mean Q-value for the child node
                mean_q_value = child.total_value / child.visits if child.visits > 0 else 0

                # Compute PUCT score
                child.puct_score = (
                    mean_q_value +
                    c_puct *
                    prior_prob *
                    math.sqrt(total_visits) / (1 + child.visits)
                )

            # Select the child with the highest PUCT score
            node = max(node.children, key=lambda child: child.puct_score)

        return node

    def _expand(self, node: "MCTSNode") -> "MCTSNode":
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

    def _evaluate(self, node: "MCTSNode") -> float:
        """
        Use the DQN model's value function to estimate the value of a leaf node.
        :param node: The leaf node to evaluate
        :return: The expected value of Q-values for the node
        """
        # Get the state representation
        price_state, process_state, gas_eua_state = self._get_state(node.env)

        with torch.no_grad():
            # Get Q-values from the DQN
            q_values = self.dqn.policy_net(price_state, process_state, gas_eua_state)
        return q_values.max().item()  # Use the maximum Q-value as the value estimate

    def _backpropagate(self, node: "MCTSNode", value: float) -> None:
        """
        Backpropagate the value from the leaf node to the root node
        :param node: The current node in the MCTS tree
        :param value: The value received from the DQN model
        """
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    def _get_state(self, env: gym.Env) -> tuple[Any, Any, Any]:
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

    def save(self, filepath: str) -> None:
        """
        Save the DQN model parameters used by MCTS_Q.
        """
        self.dqn.save(filepath)

    def load(self, filepath: str) -> None:
        """
        Load the DQN model parameters into MCTS_Q.
        """
        self.dqn.load(filepath)

    def test(self, env_config: EnvConfiguration, eps_sim_steps_test: int) -> dict[str, Any]:
        """
            Test MCTS_Q on the test environment
        """
        _, _ = self.env.reset()

        stats = np.zeros((eps_sim_steps_test, len(env_config.stats_names)))
        stats_dict_test={}

        # for i in tqdm(range(eps_sim_steps_test), desc="Testing MCTS_Q", unit="step"):
        for i in range(eps_sim_steps_test):
            # Perform step based on MCTS with DQN values
            action = self.predict(self.env)
            _, _, terminated, _, info = self.env.step(action)

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

        for m, stats_n in enumerate(env_config.stats_names):
            stats_dict_test[stats_n] = stats[:(eps_sim_steps_test), m]

        print(f"   >>Cumulative Reward {info['cum_reward']}")

        return stats_dict_test


class MCTSNode:
    """
    Represents a node in the MCTS tree.
    """
    def __init__(
            self,
            env: gym.Env,
            parent: "MCTSNode" | None = None,
            action: int | None = None,
            done: bool = False,
            remaining_steps: int = 42,
            total_steps: int = 42,
            depth: int = 0,
            maximum_depth: int = 42
        ) -> None:
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

    def is_terminal(self) -> bool:
        """
        Check if the node is terminal (i.e., if the environment is done or maximum depth is reached)
        :return: True if the node is terminal, False otherwise
        """
        return self.done or self.depth >= self.maximum_depth

    def is_fully_expanded(self) -> bool:
        """
        Check if all possible actions have been tried at this node
        :return: True if all actions have been tried, False otherwise
        """
        legal_actions = self.get_legal_actions()
        tried_actions = [child.action for child in self.children]
        return set(tried_actions) == set(legal_actions)

    def get_legal_actions(self) -> list[int]:
        """
        Get the legal actions for the current node based on the environment's action space
        :return: List of legal actions
        """
        meth_state = self.env.Meth_State  # Access the Meth_State from the environment
        if meth_state == 0:     # 'standby'
            # Allows only standby, cooldown, and startup actions
            return [0, 1, 2]
        elif meth_state == 1:   # 'cooldown'
            # Allows only standby, cooldown, and startup actions
            return [0, 1, 2]
        elif meth_state == 2:   # 'startup'
            # Allows only standby, cooldown, and load level after startup (partial load, full load)
            return [0, 1, 3, 4]
        elif meth_state == 3:   # 'partial load'
            # Allows only standby, cooldown, and load level (partial load, full load)
            return [0, 1, 3, 4]
        elif meth_state == 4:   # 'full load'
            # Allows only standby, cooldown, and load level (partial load, full load)
            return [0, 1, 3, 4]
        else:
            return list(range(self.env.action_space.n))  # Default to all actions

    def most_visited_child(self) -> "MCTSNode":
        """
        Select the child node with the most visits
        :return: The child node with the highest number of visits
        """
        return max(self.children, key=lambda child: child.visits)

class MCTSQConfiguration():
    """
    Configuration class for MCTS_Q algorithm.
    This class loads the configuration from a YAML file and initializes the MCTS_Q algorithm
    with the specified parameters.
    """

    def __init__(self) -> None:
        # Load the algorithm configuration from the YAML file
        with open("config/config_mctsq.yaml", "r", encoding="utf-8") as env_file:
            mctsq_config = yaml.safe_load(env_file)

        # Unpack data from dictionary
        self.__dict__.update(mctsq_config)

        self.rl_alg_hyp = mctsq_config

        # Initialize the string for the algorithm settings (used for file identification)
        self.str_alg = ""
        # Nested dictionary with hyperparameters, including abbreviation ('abb')
        # and variable name ('var')
        # > 'var' must match the notation in MCTS_Q/config/config_mctsq.yaml
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

    def get_hyper(self) -> str:
        """
        Displays the algorithm's hyperparameters and returns a string identifier
        for file identification.
        :return str_alg: The hyperparameter settings as a string for file identification
        """

        # Display the chosen algorithm and its hyperparameters
        print("    > MCTS_Q algorithm : >>> <<<")
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

    def hyp_print(self, hyp_name: str) -> None:
        """
        Displays the value of a specific hyperparameter and adds it to the string identifier
        for file naming
        :param hyp_name: Name of the hyperparameter to display
        """
        if hyp_name not in self.hyper:
            raise ValueError(
                f"Specified hyperparameter ({hyp_name}) is not part"
                " of the implemented settings!"
            )
        length_str = len(hyp_name)
        if length_str > 32:
            print(f"         {hyp_name} ({self.hyper[hyp_name]['abb']}):"
                  f" {self.rl_alg_hyp[self.hyper[hyp_name]['var']]}")
        elif length_str > 22:
            print(f"         {hyp_name} ({self.hyper[hyp_name]['abb']}):"
                  f"\t {self.rl_alg_hyp[self.hyper[hyp_name]['var']]}")
        elif length_str > 15:
            print(f"         {hyp_name} ({self.hyper[hyp_name]['abb']}):"
                  f"\t\t {self.rl_alg_hyp[self.hyper[hyp_name]['var']]}")
        else:
            print(f"         {hyp_name} ({self.hyper[hyp_name]['abb']}):"
                  f"\t\t\t {self.rl_alg_hyp[self.hyper[hyp_name]['var']]}")
        self.str_alg += (self.hyper[hyp_name]['abb'] +
                         str(self.rl_alg_hyp[self.hyper[hyp_name]['var']]))
