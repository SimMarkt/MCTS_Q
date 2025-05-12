import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

#TODO: Include random seeds
#TODO: Number of Hidden Layers and Activation Functions as hyperparameters

# ConvAttentionBlock
class ConvAttentionEnc(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(ConvAttentionEnc, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        self.attention_weights = nn.Conv1d(in_channels=embed_dim, out_channels=1, kernel_size=1)  # Attention scores
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim, seq_length)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        # Compute attention scores
        scores = self.attention_weights(x)  # (batch_size, 1, seq_length)
        scores = F.softmax(scores, dim=-1)  # Normalize scores across the sequence dimension

        # Apply attention weights
        x = x * scores  # Element-wise multiplication (batch_size, embed_dim, seq_length)

        # Aggregate along the sequence dimension
        x = x.sum(dim=-1)  # Weighted sum (batch_size, embed_dim)

        # Apply layer normalization
        x = self.norm(x)

        return x

# GRUAttentionBlock
class GRUAttentionEnc(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(GRUAttentionEnc, self).__init__()
        self.gru = nn.GRU(input_dim, embed_dim, batch_first=True)
        self.attention_weights = nn.Linear(embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        gru_output, _ = self.gru(x)  # (batch_size, seq_length, embed_dim)

        # Compute attention scores
        scores = self.attention_weights(gru_output).squeeze(-1)  # (batch_size, seq_length)
        scores = F.softmax(scores, dim=-1)  # Normalize scores across the sequence dimension

        # Apply attention weights
        attended = torch.bmm(scores.unsqueeze(1), gru_output).squeeze(1)  # (batch_size, embed_dim)

        # Apply layer normalization
        attended = self.norm(attended)

        return attended

# TransformerBlock
class TransformerEnc(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super(TransformerEnc, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 500, embed_dim))  # Max sequence length = 500
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        x = x + self.positional_encoding[:, :x.size(1), :]  # Add positional encoding

        # Transformer encoder
        x = self.transformer(x.permute(1, 0, 2))  # (seq_length, batch_size, embed_dim)
        x = x.mean(dim=0)  # Aggregate along the sequence dimension (batch_size, embed_dim)

        # Apply layer normalization
        x = self.norm(x)

        return x

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim, hidden_units, encoder_type="conv"):
        super(DQN, self).__init__()
        if encoder_type == "conv":
            self.encoder = ConvAttentionEnc(state_dim, embed_dim)
        elif encoder_type == "gru":
            self.encoder = GRUAttentionEnc(state_dim, embed_dim)
        elif encoder_type == "transformer":
            self.encoder = TransformerEnc(state_dim, embed_dim, num_heads=4, num_layers=2)
        else:
            raise ValueError("Invalid attention type. Choose from 'conv', 'gru', or 'transformer'.")

        self.fc_layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, state_dim, seq_length)
        x = self.encoder(x)
        x = self.fc_layers(x)
        return x

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), torch.FloatTensor(weights), indices)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

class DQNModel:
    def __init__(self, state_dim, action_dim, seq_length, embed_dim, hidden_units, buffer_capacity, batch_size, gamma, lr, epsilon_start, epsilon_end, epsilon_decay, encoder_type="conv"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_length = seq_length
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy_net = DQN(state_dim, action_dim, embed_dim, hidden_units, encoder_type)
        self.target_net = DQN(state_dim, action_dim, embed_dim, hidden_units, encoder_type)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def compute_multi_step_return(self, rewards, next_states, dones, n_steps):
        """Compute multi-step returns."""
        returns = []
        for t in range(len(rewards)):
            G = 0
            discount = 1
            for k in range(n_steps):
                if t + k < len(rewards):
                    G += discount * rewards[t + k]
                    discount *= self.gamma
                    if dones[t + k]:
                        break
            returns.append(G)
        return torch.FloatTensor(returns)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values with multi-step returns
        n_steps = 3  # Example: 3-step returns
        multi_step_returns = self.compute_multi_step_return(rewards, next_states, dones, n_steps)
        with torch.no_grad():
            next_q_values_policy = self.policy_net(next_states).max(1)[0]
            next_q_values_target = self.target_net(next_states).max(1)[0]
            next_q_values = 0.5 * (next_q_values_policy + next_q_values_target)  # Target regularization
            target_q_values = multi_step_returns + (self.gamma ** n_steps) * next_q_values * (1 - dones)

        # Compute loss with importance sampling weights
        td_error = q_values - target_q_values
        loss = (weights * td_error.pow(2)).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)  # Gradient clipping
        self.optimizer.step()

        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_error.abs().detach().numpy())

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())