import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time, itertools

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    if x.ndim == 1:
        x = x.reshape(1, -1) # Convert 1D array to 2D array
    e_x = np.exp(x)
    return e_x / (np.sum(e_x, axis=1, keepdims=True) + 1e-16) # Add small value to avoid division by zero

# Define system dynamics
def system_dynamics(x, u, dt=1):
    return x + u * dt

# Define observation model
def observation_model(x):
    return x

def custom_cdist(x, goals, types):
    """
    Compute the pairwise distance between rows of x and rows of goals based on measurement types.
    """
    diff_to_goals = x[:, np.newaxis, :] - goals[np.newaxis, :, :]
    distances_to_goals = np.linalg.norm(diff_to_goals, axis=2)
    angles_to_goal = np.arctan2(diff_to_goals[:, :, 1], diff_to_goals[:, :, 0])
    diff_to_robot = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    angles_to_robot = np.arctan2(diff_to_robot[:, :, 1], diff_to_robot[:, :, 0])
    relative_angles = np.abs((angles_to_goal[np.newaxis, :, :] - angles_to_robot[:, :, np.newaxis] + np.pi) % (2 * np.pi) - np.pi)
    alignment = 1 - np.cos(relative_angles)
    return distances_to_goals

def calculate_joint_goal_probs(agent_poses, goals, predict_types, reward_configs, eta=10):
    num_agents = agent_poses.shape[0]
    num_goals = goals.shape[0]
    distances = custom_cdist(agent_poses, goals, predict_types)
    evidence = eta * np.exp(-1.0 / eta * distances)
    probabilities = softmax(evidence)
    joint_probabilities = np.ones([num_goals] * num_agents, dtype=float)
    for i in range(num_agents):
        joint_probabilities *= probabilities[i].reshape([num_goals if j == i else 1 for j in range(num_agents)])
    likelihood = np.array([joint_probabilities[tuple(config)] for config in reward_configs], dtype=np.float64)
    likelihood = softmax(likelihood)
    return likelihood, distances

def compute_distance_rhc(x, goals):
    diff = x[:, np.newaxis, :] - goals[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    return distances

def compute_entropy_rhc(evidence, prior):
    likelihood = evidence
    posterior = softmax(likelihood * prior)
    total_cost = -np.dot(posterior, np.log(posterior.T)).reshape(-1).astype(np.float64)
    return total_cost, posterior

def compute_cost_rhc(x, goals, prior, types, reward_configs):
    likelihood, distances = calculate_joint_goal_probs(x, goals, types, reward_configs)
    total_cost, posterior = compute_entropy_rhc(likelihood, prior)
    return total_cost * 1e1, posterior, distances

def objective(u, x_init, goals, N, prior, dt=1, lambda_reg=0.001, alpha_reg=0.001, reward_configs=None, measurement_types=None):
    total_free_energy = 0
    num_agents = x_init.shape[0]
    u = u.reshape(num_agents, N, 2)
    x_curr = x_init.copy()
    for k in range(N):
        u_curr = u[:, k]
        x_next = system_dynamics(x_curr, u_curr, dt)
        free_energy, prior, distances = compute_cost_rhc(x_next, goals, prior, measurement_types, reward_configs)
        total_free_energy += free_energy
        total_free_energy += lambda_reg * (np.sum(u_curr ** 2))
        min_goal = np.argmin(prior)
        total_free_energy += alpha_reg * np.sum([distances[i, min_goal] for i in range(num_agents)])
        x_curr = x_next
    return total_free_energy

# Define a function to run the real-time simulation
def run_rhc(initial_state, agent_vars):
    N = agent_vars['horizon']
    dt = 1#agent_vars['dt']
    lambda_reg = 0.001
    alpha_reg = 0.001
    reward_configs = agent_vars['reward_configs']
    measurement_types = agent_vars['agent_types']
    goals = agent_vars['goals']

    # Initialize the control inputs
    x_init = initial_state[:,:2]
    num_agents = x_init.shape[0]
    u_init = np.random.rand(num_agents, N, 2)
    prior = np.ones(len(reward_configs)) / len(reward_configs)
    # Optimize the control inputs
    result = minimize(objective, u_init.flatten(), args=(x_init, goals, N, prior, dt, lambda_reg, alpha_reg, reward_configs, measurement_types), 
                        bounds=[(-1, 1), (-1, 1)] * N * num_agents, method='L-BFGS-B', options={'maxiter': 500})
    optimal_u = result.x.reshape(num_agents, N, 2)
    best_value = result.fun
    return optimal_u, best_value