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

class MCTSNode:
    def __init__(self, env, parent=None, action=None, done=False, remaining_steps=42, total_steps=42, depth=0, maximum_depth=42):
        self.env = env
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
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
        
    def best_child(self, c_param=1.41):
        """
        Select the best child node based on UCT (Upper Confidence Bound for Trees)
        :param c_param: Exploration parameter for UCT
        :return: The child node with the highest weighted score
        """
        choices_weights = [
            (child.total_reward / child.visits) + 
            c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def most_visited_child(self):
        """
        Select the child node with the most visits
        :return: The child node with the highest number of visits
        """
        return max(self.children, key=lambda child: child.visits)
     
class MCTS:
    def __init__(self):
        # Load the algorithm configuration from the YAML file
        with open("config/config_mcts.yaml", "r") as env_file:
            mcts_config = yaml.safe_load(env_file)

        # Unpack data from dictionary
        self.__dict__.update(mcts_config)

        # Initialize dictionary for MCTS run results
        self.stats_dict_test = {}

        # Initialize variables for storing tree structure (for debugging purposes)
        self.path = None            # System path
        self.time_store = False  # Flag to store the tree structure
        self.Meth_State = None
        self.init_el_price = None
        self.init_pot_reward = None
        self.step = None

        self.tree_log = []  # List to store tree structure

    def search(self, root_env, Meth_State=None, init_el_price=None, init_pot_reward=None):
        """
        Perform MCTS search to find the best action
        :param root_env: The initial environment to start the search
        :param Meth_State: The current state of the Power-to-Gas process
        :param init_el_price: The initial electricity price
        :param init_pot_reward: The initial potential reward
        :return: The best action found by MCTS (the action with the highest visit count)
        """
        self.Meth_State = Meth_State
        self.init_el_price = init_el_price
        self.init_pot_reward = init_pot_reward

        root_env_copy = copy.deepcopy(root_env)
        root_node = MCTSNode(root_env_copy, total_steps=self.total_steps, maximum_depth=self.maximum_depth)

        for _ in range(self.iterations):
            node = self._select(root_node)
            if not node.is_terminal():
                node = self._expand(node)
            reward = self._simulate(node.env, node.done, node.action, node.remaining_steps)
            self._backpropagate(node, reward)

        # Log the tree structure and save it to a CSV file if time_store is True
        if self.time_store:
            self._log_tree_structure(root_node)
            self._save_tree_to_csv()
            self.time_store = False  # Reset the flag after saving

        return root_node.most_visited_child().action

    def _select(self, node):
        """
        Select the best child node based on UCT (Upper Confidence Bound for Trees)
        :param node: The current node in the MCTS tree
        :return: The selected child node
        """
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        return node

    def _expand(self, node):
        """
        Expand the node by adding a new child node
        :param node: The current node in the MCTS tree
        :return: The newly created child node
        """
        if node.depth >= self.maximum_depth:  # Prevent expansion beyond max depth
            return node

        legal_actions = node.get_legal_actions()
        tried_actions = [child.action for child in node.children]
        untried_actions = [a for a in legal_actions if a not in tried_actions]

        action = random.choice(untried_actions)
        new_env = copy.deepcopy(node.env)
        obs, reward, terminated, truncated, info = new_env.step(action)
        done = terminated or truncated
        remaining_steps = node.remaining_steps - 1  # Decrease remaining steps by 1 each time we expand (To keep the simulation time equal)
        child_node = MCTSNode(new_env, parent=node, action=action, done=done, remaining_steps=remaining_steps, total_steps=self.total_steps, maximum_depth=self.maximum_depth)
        node.children.append(child_node)
        return child_node

    def _simulate(self, env, done, action, remaining_steps):
        """
        Simulate the environment from the current node to a leaf node
        :param env: The environment to simulate
        :param done: Whether the environment is done
        :param action: The action taken at the current node
        :param remaining_steps: The number of steps remaining for the simulation
        :return: The total reward received during the simulation
        """
        sim_env = copy.deepcopy(env)
        total_reward = 0

        step_count = 0
        while not done and step_count < remaining_steps:
            # Perform the same action as the expanded node during rollout
            _, reward, terminated, truncated, _ = sim_env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

        return total_reward

    def _backpropagate(self, node, reward):
        """
        Backpropagate the reward from the leaf node to the root node
        :param node: The current node in the MCTS tree
        :param reward: The reward received from the simulation
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def store_tree(self, step):
        """
        Set the flag to store the tree structure
        :param step: The current step in testing
        """
        self.time_store = True
        self.step = step

    def _log_tree_structure(self, node):
        """
        Recursively log the tree structure
        :param node: The current node in the MCTS tree
        """
        node_id = id(node)
        self.tree_log.append({
            "node_id": node_id,
            "parent_id": id(node.parent) if node.parent else None,
            "action": node.action,
            "depth": node.depth,
            "visits": node.visits,
            "total_reward_p_visits": node.total_reward/node.visits,
            "Meth_State": self.Meth_State,
            "init_el_price": self.init_el_price,
            "init_pot_reward": self.init_pot_reward,
        })
        for child in node.children:
            self._log_tree_structure(child)

    def _save_tree_to_csv(self):
        """
        Save the logged tree structure to a CSV file
        """
        with open(self.path + self.path_log + f"tree_structure_step{self.step}.csv", "w", newline="") as csvfile:
            fieldnames = ["node_id", "parent_id", "action", "depth", "visits", "total_reward_p_visits", "Meth_State", "init_el_price", "init_pot_reward"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.tree_log)

    def run(self, env, EnvConfig, timesteps):
        """
        Run MCTS on the test environment.
        :param env: The test environment
        :param EnvConfig: The environment configuration object
        :param timesteps: The number of timesteps to run
        """
        obs, _ = env.reset()
        stats = np.zeros((timesteps, len(EnvConfig.stats_names)))
        pot_reward = 0

        timesteps = 2900

        for i in tqdm(range(timesteps), desc='---Apply MCTS planning on the test environment:'):
            
            if i % self.store_interval == 0 and i != 0: self.store_tree(i)  # Store the tree structure every store_interval steps

            action = self.search(env, Meth_State=obs['METH_STATUS'], init_el_price=obs['Elec_Price'][0], init_pot_reward=pot_reward)  # Perform MCTS search to get the best action
            
            obs, _ , _ , terminated, info = env.step(action)
            pot_reward = info['Pot_Reward']
            print(f' Pot_Rew {pot_reward/6}, Load_Id {info["Part_Full"]}, Meth_State {info["Meth_State"]}, Rew {info["reward [ct]"]}, Action {action}')
            print(f"Scenario {EnvConfig.scenario}, Iterations {self.iterations}, exploration_weight {self.exploration_weight}, total_steps {self.total_steps}, maximum_depth {self.maximum_depth}")

            # Store data in stats
            if not terminated:
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
            self.stats_dict_test[EnvConfig.stats_names[m]] = stats[:(timesteps), m]
