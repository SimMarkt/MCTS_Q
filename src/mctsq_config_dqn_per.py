"""
----------------------------------------------------------------------------------------------------
MCTS_Q: Monte Carlo Tree Search with Deep-Q-Network
GitHub Repository: https://github.com/SimMarkt/MCTS_Q

mctsq_config_dqn_per: 
> Provides the Deep Q-Network model including different encoders for energy market and process data
> Implements a prioritized replay buffer for experience replay
----------------------------------------------------------------------------------------------------
"""
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- ConvAttentionEnc ---
class ConvAttentionEnc(nn.Module):
    """Convolutional Neural Network with Attention Mechanism for Encoding Time-Series Data."""
    def __init__(self, input_dim, embed_dim):
        super(ConvAttentionEnc, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=embed_dim,
            kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            padding=1
        )
        self.attention_weights = nn.Conv1d(in_channels=embed_dim, out_channels=1, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """ 
        Forward pass through the ConvAttentionEnc.
        :param x: Input tensor of shape (batch_size, seq_length, input_dim)
        :return: Encoded tensor of shape (batch_size, embed_dim)
        """
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_length)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        scores = self.attention_weights(x)  # (batch_size, 1, seq_length)
        scores = F.softmax(scores, dim=-1)
        x = x * scores
        x = x.sum(dim=-1)
        x = self.norm(x)
        return x

# --- GRUAttentionEnc ---
class GRUAttentionEnc(nn.Module):
    """GRU-based Encoder with Attention Mechanism for Time-Series Data."""
    def __init__(self, input_dim, embed_dim):
        super(GRUAttentionEnc, self).__init__()
        self.gru = nn.GRU(input_dim, embed_dim, batch_first=True)
        self.attention_weights = nn.Linear(embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass through the GRUAttentionEnc.
        :param x: Input tensor of shape (batch_size, seq_length, input_dim)
        :return: Encoded tensor of shape (batch_size, embed_dim)
        """
        gru_output, _ = self.gru(x)
        scores = self.attention_weights(gru_output).squeeze(-1)
        scores = F.softmax(scores, dim=-1)
        attended = torch.bmm(scores.unsqueeze(1), gru_output).squeeze(1)
        attended = self.norm(attended)
        return attended

# --- TransformerEnc ---
class TransformerEnc(nn.Module):
    """Transformer Encoder for Time-Series Data."""
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super(TransformerEnc, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 500, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass through the TransformerEnc.
        :param x: Input tensor of shape (batch_size, seq_length, input_dim)
        :return: Encoded tensor of shape (batch_size, embed_dim)
        """
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x.permute(1, 0, 2))
        x = x.mean(dim=0)
        x = self.norm(x)
        return x

# --- GasEUAEncoder ---
class GasEUAEncoder(nn.Module):
    """Encoder for Gas and EUA Data, supporting MLP or GRU-based architectures."""
    def __init__(self, input_dim, embed_dim, encoder_type="mlp"):
        super(GasEUAEncoder, self).__init__()
        if encoder_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(input_dim * 3, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU()
            )
            self.flatten = True
        elif encoder_type == "gru":
            self.encoder = GRUAttentionEnc(input_dim, embed_dim)
            self.flatten = False
        else:
            raise ValueError("Invalid gas/EUA encoder type.")

    def forward(self, x):
        """
        Forward pass through the GasEUAEncoder.
        :param x: Input tensor of shape (batch_size, seq_length, input_dim)
        :return: Encoded tensor of shape (batch_size, embed_dim)
        """
        if self.flatten:
            x = x.view(x.size(0), -1)
        return self.encoder(x)

def get_activation(activation_name):
    """
    Returns the activation function based on the provided name in config.mctsq.yaml.
    :param activation_name: Name of the activation function (e.g., "relu", "tanh")
    :return: Corresponding PyTorch activation function
    """
    if activation_name.lower() == "relu":
        return nn.ReLU()
    elif activation_name.lower() == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

# --- TripleEncoderDQN ---
class TripleEncoderDQN(nn.Module):
    """
    DQN with three parallel encoders for different time-series modalities.
    """
    def __init__(
            self,
            el_input_dim, process_input_dim, gas_eua_input_dim,  
            embed_dim, hidden_layers, hidden_units, action_dim,
            price_encoder_type="conv", process_encoder_type="gru", gas_eua_encoder_type="mlp",
            activation="relu"
        ):
        """
        :param el_input_dim: Input dimension for electricity price data
        :param process_input_dim: Input dimension for process data
        :param gas_eua_input_dim: Input dimension for gas and EUA data
        :param embed_dim: Embedding dimension for each encoder
        :param hidden_layers: Number of hidden layers in the fully connected network
        :param hidden_units: Number of units in each hidden layer
        :param action_dim: Number of possible actions
        :param price_encoder_type: Type of encoder for electricity price data
                                   ("conv", "gru", "transformer")
        :param process_encoder_type: Type of encoder for process data ("conv", "gru", "transformer")
        :param gas_eua_encoder_type: Type of encoder for gas and EUA data ("mlp", "gru")
        :param activation: Activation function for the fully connected layers ("relu", "tanh")
        """
        super(TripleEncoderDQN, self).__init__()

        # Electricity Price encoder
        if price_encoder_type == "conv":
            self.price_encoder = ConvAttentionEnc(el_input_dim, embed_dim)
        elif price_encoder_type == "gru":
            self.price_encoder = GRUAttentionEnc(el_input_dim, embed_dim)
        elif price_encoder_type == "transformer":
            self.price_encoder = TransformerEnc(el_input_dim, embed_dim, num_heads=4, num_layers=2)
        else:
            raise ValueError("Invalid price encoder type.")

        # Process encoder
        if process_encoder_type == "conv":
            self.process_encoder = ConvAttentionEnc(process_input_dim, embed_dim)
        elif process_encoder_type == "gru":
            self.process_encoder = GRUAttentionEnc(process_input_dim, embed_dim)
        elif process_encoder_type == "transformer":
            self.process_encoder = TransformerEnc(process_input_dim, embed_dim,
                                                  num_heads=4, num_layers=2)
        else:
            raise ValueError("Invalid process encoder type.")

        # Gas/EUA encoder
        self.gas_eua_encoder = GasEUAEncoder(gas_eua_input_dim, embed_dim,
                                             encoder_type=gas_eua_encoder_type)

        # Build fully connected layers dynamically
        fc_layers = []
        input_dim = embed_dim * 3
        act = get_activation(activation)
        for i in range(hidden_layers):
            fc_layers.append(nn.Linear(int(input_dim), int(hidden_units)))
            fc_layers.append(act)
            input_dim = hidden_units
        fc_layers.append(nn.Linear(int(hidden_units), int(action_dim)))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, price_data, process_data, gas_eua_data):
        """
        Forward pass through the TripleEncoderDQN.
        :param price_data: Electricity price data tensor of shape
                           (batch_size, seq_length, el_input_dim)
        :param process_data: Process data tensor of shape
                             (batch_size, seq_length, process_input_dim)
        :param gas_eua_data: Gas and EUA data tensor of shape
                             (batch_size, seq_length, gas_eua_input_dim)
        :return: Q-values tensor of shape (batch_size, action_dim)
        """
        price_feat = self.price_encoder(price_data)
        process_feat = self.process_encoder(process_data)
        gas_eua_feat = self.gas_eua_encoder(gas_eua_data)
        combined = torch.cat([price_feat, process_feat, gas_eua_feat], dim=-1)
        return self.fc_layers(combined)

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer for DQN."""
    def __init__(self, capacity, alpha=0.6):
        """
        :param capacity: Maximum number of experiences to store
        :param alpha: Prioritization exponent (0 = no prioritization, 1 = full prioritization)
        """
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer with maximum priority.
        :param state: Current state
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state after action
        :param done: Whether the episode has ended
        """
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences based on their priorities.
        :param batch_size: Number of experiences to sample
        :param beta: Importance-sampling exponent (0 = no corrections, 1 = full correction)
        :return: Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        priorities = np.array(self.priorities)
        # Raise priorities to the power of alpha to emphasize higher priority samples
        # (alpha = 0: uniform sampling, alpha > 0: prioritize important samples)
        probabilities = priorities ** self.alpha
        # Divide all probabilities by their sum to normalize them,
        # and provide a probability distribution
        probabilities /= probabilities.sum()

        # Randomly sample indices based on the probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        # len(self.buffer) * probabilities[indices]:
        # The expected number of times each sampled transition would be selected.
        # If sampling were uniform, each probability would be 1/len(self.buffer),
        # so this product would be 1.
        # Calculate importance sampling weights, where beta is the degree of correction
        # for the bias introduced by prioritized sampling
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()        # Normalize weights to prevent large updates (for stability)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), torch.FloatTensor(weights), indices)

    def update_priorities(self, indices, priorities):
        """
        Update the priorities of sampled experiences.
        :param indices: List of indices of the experiences to update
        :param priorities: New priority values for the experiences
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        """
        Return the current size of the buffer.
        :return: Number of experiences in the buffer
        """
        return len(self.buffer)

class DQNModel:
    """
    Deep Q-Network model with three encoders for different time-series modalities.
    """
    def __init__(
            self,
            el_input_dim, process_input_dim, gas_eua_input_dim, 
            action_dim, embed_dim, hidden_layers, hidden_units, buffer_capacity, batch_size, gamma, lr,
            price_encoder_type="conv", process_encoder_type="gru", gas_eua_encoder_type="mlp",
            activation="relu", learning_starts=10000, seed=None
        ):
        """
        :param el_input_dim: Input dimension for electricity price data
        :param process_input_dim: Input dimension for process data
        :param gas_eua_input_dim: Input dimension for gas and EUA data
        :param action_dim: Number of possible actions
        :param embed_dim: Embedding dimension for each encoder
        :param hidden_layers: Number of hidden layers in the fully connected network
        :param hidden_units: Number of units in each hidden layer
        :param buffer_capacity: Maximum number of experiences to store in the replay buffer
        :param batch_size: Batch size for training
        :param gamma: Discount factor for future rewards
        :param lr: Learning rate for the optimizer
        :param price_encoder_type: Type of encoder for electricity price data 
                                   ("conv", "gru", "transformer")
        :param process_encoder_type: Type of encoder for process data ("conv", "gru", "transformer")
        :param gas_eua_encoder_type: Type of encoder for gas and EUA data ("mlp", "gru")
        :param activation: Activation function for the fully connected layers ("relu", "tanh")
        :param learning_starts: Number of experiences to collect before starting training
        :param seed: Random seed for reproducibility
        """
        # Set random seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.el_input_dim = el_input_dim
        self.gas_eua_input_dim = gas_eua_input_dim
        self.process_input_dim = process_input_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts

        self.policy_net = TripleEncoderDQN(
            el_input_dim, process_input_dim, gas_eua_input_dim,
            embed_dim, hidden_layers, hidden_units, action_dim,
            price_encoder_type, process_encoder_type, gas_eua_encoder_type,
            activation
        )
        self.target_net = TripleEncoderDQN(
            el_input_dim, process_input_dim, gas_eua_input_dim,
            embed_dim, hidden_layers, hidden_units, action_dim,
            price_encoder_type, process_encoder_type, gas_eua_encoder_type,
            activation
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)

    def update(self):
        """
        Update the policy network using a batch from the replay buffer.
        :return: Loss value or None if not enough samples
        """
        cond1 = len(self.replay_buffer) < self.batch_size
        cond2 = len(self.replay_buffer) < self.learning_starts
        if cond1 or cond2:
            return None

        # Unpack buffer: states should be tuples (price_state, process_state)
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, weights, indices = batch

        price_states = np.array([s["Elec_Price"] for s in states])[..., np.newaxis]
        process_states = np.array([
            np.stack(
                [
                    s["T_CAT"], s["H2_in_MolarFlow"], s["CH4_syn_MolarFlow"],
                    s["H2_res_MolarFlow"], s["H2O_DE_MassFlow"], s["Elec_Heating"]
                ],
                axis=-1)
            for s in states
        ])
        gas_eua_states = np.array([
            np.stack([s["Gas_Price"], s["EUA_Price"]], axis=-1)
            for s in states
        ])
        next_price_states = np.array([s["Elec_Price"] for s in next_states])[..., np.newaxis]
        next_process_states = np.array([
            np.stack(
                [
                    s["T_CAT"], s["H2_in_MolarFlow"], s["CH4_syn_MolarFlow"],
                    s["H2_res_MolarFlow"], s["H2O_DE_MassFlow"], s["Elec_Heating"]
                ],
                axis=-1)
            for s in next_states
        ])
        next_gas_eua_states = np.array([
            np.stack([s["Gas_Price"], s["EUA_Price"]], axis=-1)
            for s in next_states
        ])

        price_states = torch.FloatTensor(price_states)
        process_states = torch.FloatTensor(process_states)
        gas_eua_states = torch.FloatTensor(gas_eua_states)
        next_price_states = torch.FloatTensor(next_price_states)
        next_process_states = torch.FloatTensor(next_process_states)
        next_gas_eua_states = torch.FloatTensor(next_gas_eua_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Compute Q values
        q_values = self.policy_net(
            price_states, process_states, gas_eua_states
            ).gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- One-step TD target update ---
        with torch.no_grad():
            next_q_values = self.target_net(
                next_price_states, next_process_states, next_gas_eua_states
                ).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        td_error = q_values - target_q_values
        loss = (weights * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_error.abs().detach().numpy())

        return loss

    def save(self, filepath):
        """
        Save the policy network, target network, optimizer, and replay buffer.
        :param filepath: Path to save the model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer_data': {
                'buffer': list(self.replay_buffer.buffer),
                'priorities': list(self.replay_buffer.priorities),
                'alpha': self.replay_buffer.alpha,
                'capacity': self.replay_buffer.buffer.maxlen
            }
        }, filepath)

    def load(self, filepath):
        """
        Load the policy network, target network, optimizer, epsilon, and replay buffer.
        :param filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'replay_buffer_data' in checkpoint:
            data = checkpoint['replay_buffer_data']
            self.replay_buffer = PrioritizedReplayBuffer(data['capacity'], alpha=data['alpha'])
            self.replay_buffer.buffer = deque(data['buffer'], maxlen=data['capacity'])
            self.replay_buffer.priorities = deque(data['priorities'], maxlen=data['capacity'])

    def update_target_network(self, tau=0.005):
        """
        Polyak (soft) update for the target network.
        :param tau: Interpolation parameter for soft update (0 < tau <= 1)
        """
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
