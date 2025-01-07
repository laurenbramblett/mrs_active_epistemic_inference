import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class CustomEnv:
    def __init__(self):
        self.state = None
        self.action_space = 1  # Example: One continuous action dimension
        self.observation_space = 3  # Example: Three-dimensional state

    def reset(self):
        self.state = np.random.rand(3)
        return self.state

    def step(self, action):
        # Assume the action is continuous and affects the state linearly
        self.state = self.state + action * np.random.rand(3)
        reward = -np.sum(np.square(self.state))  # Example: Reward is negative distance from origin
        done = np.linalg.norm(self.state) > 10  # Example: Episode ends if state is far from origin
        return self.state, reward, done, {}

    def render(self):
        pass

    def close(self):
        pass

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=0.003, gamma=0.99, tau=0.005, alpha=0.2):
        self.actor = PolicyNetwork(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1 = SoftQNetwork(state_dim, action_dim).to(device)
        self.critic2 = SoftQNetwork(state_dim, action_dim).to(device)
        self.critic1_target = SoftQNetwork(state_dim, action_dim).to(device)
        self.critic2_target = SoftQNetwork(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_target = self.critic1_target(next_state, next_action)
            q2_target = self.critic2_target(next_state, next_action)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.gamma * q_target

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        critic1_loss = nn.MSELoss()(q1, q_target)
        critic2_loss = nn.MSELoss()(q2, q_target)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

env = CustomEnv()
state_dim = env.observation_space
action_dim = env.action_space  # Assuming a single continuous action dimension
max_action = 1.0  # Assuming action range is [-1, 1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = SACAgent(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(capacity=10000)

num_episodes = 1000
batch_size = 64

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        if len(replay_buffer) > batch_size:
            agent.train(replay_buffer, batch_size)

    if episode % 100 == 0:
        print(f"Episode {episode} completed")

env.close()
