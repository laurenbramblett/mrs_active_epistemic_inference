import numpy as np
import gymnasium as gym
from gymnasium import spaces

def softmax(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / (np.sum(e_x, axis=1, keepdims=True) + 1e-16)

class MAMGEnv(gym.Env):
    def __init__(self):
        super(MAMGEnv, self).__init__()
        self.num_agents = 2
        self.env_size = 3
        self.num_goals = 2
        self.goals = np.array([[1,1.1], [0,1]]) * self.env_size #np.random.rand(self.num_goals, 2) * self.env_size
        self.state_size = self.num_agents * 2
        self.action_size = 2
        self.lambda_reg = 0.001
        self.alpha_reg = 0.001
        self.prior = np.zeros(self.num_goals) + 1.0 / self.num_goals
        self.convergence_reward = 100
        self.reward_configs = [tuple(np.repeat(i, self.num_agents)) for i in range(self.num_goals)]
        self.total_cost = 0

        self.observation_space = spaces.Box(low=0, high=self.env_size, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_agents, self.action_size), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = np.array([[0,0],[1,0]])*self.env_size#np.random.rand(self.num_agents, 2) * self.env_size
        self.timesteps = 0
        self.total_cost = 0
        self.prior = np.zeros(self.num_goals) + 1.0 / self.num_goals
        return (self.positions.flatten().astype(np.float32), {})

    def step(self, action):
        actions = action.reshape((self.num_agents, 2))
        self.positions = np.clip(self.positions + actions, 0, self.env_size)
        self.timesteps += 1

        free_energy_cost, distances = self._compute_cost(self.positions, self.goals)
        movement_cost = self.lambda_reg * np.sum(actions**2)
        min_goal = np.argmin(self.prior)
        belief_goal_dist_cost = self.alpha_reg * np.sum([distances[i,min_goal] for i in range(self.num_agents)])
        total_cost = free_energy_cost + movement_cost + belief_goal_dist_cost

        self.total_cost += total_cost[0]

        terminated = bool(self.timesteps >= 10)
        truncated = False
        obs = self.positions.flatten().astype(np.float32)
        
        reward = -self.total_cost
        return obs, reward, terminated, truncated, {}
    
    def custom_cdist(self, x, goals, types = None):
        """
        Compute the pairwise distance between rows of x1 and rows of x2 based on measurement types.

        Args:
            x (np.ndarray): An array of shape (m, d)
            goals (np.ndarray): An array of shape (n, d)
            types (list): A list of measurement types for each pair of rows.
        Returns:
            np.ndarray: An array of shape (m, n) with the pairwise evidences.
        """
        # assert len(types) == x.shape[0], "Length of types must match number of rows in x"

        # Compute the pairwise differences to goals
        diff_to_goals = x[:, np.newaxis, :] - goals[np.newaxis, :, :]
        
        # Compute the pairwise distances to goals
        distances_to_goals = np.linalg.norm(diff_to_goals, axis=2)
        
        # Compute the pairwise angles to goals
        angles_to_goal = np.arctan2(diff_to_goals[:, :, 1], diff_to_goals[:, :, 0])
        
        # Compute the pairwise differences between robots
        diff_to_robot = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        
        # Compute the pairwise angles between robots
        angles_to_robot = np.arctan2(diff_to_robot[:, :, 1], diff_to_robot[:, :, 0])
        
        # Compute the relative angles
        relative_angles = np.abs((angles_to_goal[np.newaxis, :, :] - angles_to_robot[:, :, np.newaxis] + np.pi) % (2 * np.pi) - np.pi)
        
        # Using cosine to value alignment, where 1 means perfectly aligned and -1 means opposite
        alignment = 1 - np.cos(relative_angles)
        # TODO: Utilize alignment for different type of bot

        return distances_to_goals

    def calculate_joint_goal_probs(self,agent_poses, goals, predict_types = None, reward_configs = None, eta = 10):
        """
        Calculate the joint goal probabilities for any number of agents and goals,
        applying a reward to specified configurations.

        Parameters:
        - agent_poses (np.ndarray): Array of shape [num_agents, 2] representing the positions of agents.
        - goals (np.ndarray): Array of shape [num_goals, 2] representing the positions of goals.
        - predict_types (list): List of types for prediction
        - reward_configs (list of tuples): List of configurations to reward. Each configuration is a tuple of goal indices.
        
        Returns:
        - joint_probabilities (np.ndarray): Array representing the joint probabilities.
        """
        num_agents = agent_poses.shape[0]
        num_goals = goals.shape[0]

        # Calculate distances between agents and goals
        distances = self.custom_cdist(agent_poses, goals)
        evidence = eta * np.exp(-1.0 / eta * distances)

        # Convert distances to probabilities using softmax
        probabilities = softmax(evidence) # Apply softmax along the goal dimension

        # Initialize joint probabilities as an array of ones with the appropriate shape
        joint_probabilities = np.ones([num_goals] * num_agents, dtype=float)

        # Calculate joint probabilities
        for i in range(num_agents):
            joint_probabilities *= probabilities[i].reshape([num_goals if j == i else 1 for j in range(num_agents)]) 

        # Only return the specified configurations
        likelihood = np.array([joint_probabilities[tuple(config)] for config in self.reward_configs], dtype=np.float64)

        # Normalize the joint probabilities
        likelihood = softmax(likelihood)

        return likelihood

    def _compute_distance(self, x, goals):
        diff = x[:, np.newaxis, :] - goals[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        return distances

    def _compute_evidence(self, x, goals, eta=10):
        distances = self._compute_distance(x, goals)
        return eta * np.exp(-1.0 / eta * distances), distances

    def _compute_entropy(self, evidence):
        sum_evidence = np.sum(evidence, axis=0)
        max_sum_evidence = np.max(sum_evidence)
        likelihood = np.exp(-(sum_evidence - max_sum_evidence)) / (np.sum(np.exp(-(sum_evidence - max_sum_evidence)) + 1e-16))
        # likelihood = np.exp(-sum_evidence) / np.sum(np.exp(-sum_evidence))
        self.posterior = softmax(likelihood * self.prior)
        total_cost = -np.dot(self.posterior, np.log(self.posterior.T)).reshape(-1).astype(np.float64)
        return total_cost

    def _compute_cost(self, x, goals):
        # evidence = self.calculate_joint_goal_probs(x, goals)
        evidence, distances = self._compute_evidence(x, goals)
        return self._compute_entropy(evidence) * 1e1, distances

    def _check_convergence(self, threshold=0.1):
        distances_to_goals = self._compute_distance(self.positions, self.goals)
        distances_to_selected_goal = [np.min(distances) for distances in distances_to_goals]
        selected_goals = [np.argmin(distances) for distances in distances_to_goals]
        all_same_goal = [selected_goals[0] == which_goal for which_goal in selected_goals]
        check = (np.array(distances_to_selected_goal) < threshold).all() and all(all_same_goal)
        return check
