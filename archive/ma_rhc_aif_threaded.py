import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from ray import tune
from ray.rllib.agents.maddpg import MADDPGTrainer

# Define the multi-agent environment
class MultiAgentEnv(gym.Env):
    def __init__(self, config):
        self.num_agents = 3
        self.env_size = 6
        self.num_goals = 2
        self.goals = np.random.rand(self.num_goals, 2) * self.env_size
        self.state_size = self.num_agents * 2
        self.action_size = 2

        self.observation_space = spaces.Box(low=0, high=self.env_size, shape=(self.state_size,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_size,))
        
        self.reset()

    def reset(self):
        self.positions = np.random.rand(self.num_agents, 2) * self.env_size
        self.timesteps = 0
        return self._get_obs()

    def step(self, actions):
        actions = actions.reshape(self.num_agents, 2)
        self.positions += actions
        self.timesteps += 1

        rewards = -self._compute_free_energy()
        done = self.timesteps >= 50
        
        return self._get_obs(), rewards, done, {}

    def _get_obs(self):
        return self.positions.flatten()

    def _compute_free_energy(self):
        distances = np.linalg.norm(self.positions[:, np.newaxis] - self.goals[np.newaxis], axis=2)
        evidence = np.exp(-distances)
        entropy = -np.sum(evidence * np.log(evidence + 1e-10), axis=1)
        return entropy

# Configuration for RLlib
config = {
    "env": MultiAgentEnv,
    "env_config": {},
    "multiagent": {
        "policies": {
            f"agent_{i}": (None, spaces.Box(low=0, high=6, shape=(6,)), spaces.Box(low=-1, high=1, shape=(2,)), {}) for i in range(3)
        },
        "policy_mapping_fn": lambda agent_id: f"agent_{int(agent_id[-1])}"
    },
    "framework": "torch",
    "num_gpus": 0,
    "num_workers": 1,
}

# Train the multi-agent policy
trainer = MADDPGTrainer(config=config)
for i in range(100):
    result = trainer.train()
    print(f"Iteration {i}: mean reward = {result['episode_reward_mean']}")

# Evaluate the trained policy
env = MultiAgentEnv({})
obs = env.reset()
for _ in range(50):
    actions = [trainer.compute_action(obs[i], policy_id=f"agent_{i}") for i in range(3)]
    obs, rewards, done, _ = env.step(np.array(actions))
    if done:
        break

# Plot the resulting trajectories
trajectories = env.positions
for i in range(3):
    plt.plot(trajectories[:, i, 0], trajectories[:, i, 1], marker='o', label=f'Trajectory Agent {i+1}')
plt.scatter(env.goals[:,0], env.goals[:,1], marker='x', color='red', label='Goals', s=100)
plt.title('Agent Trajectories to Goals')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid()
plt.show()
