import numpy as np
from matplotlib.patches import Polygon, Circle
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set1_9
import copy, time, json, sys # torch
import pandas as pd
from mcts_class import MCTSNode
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

#------------------------------------------------------------------------------------------
# ----------------- Functions for sofmax and stable log calculations ----------------------
#------------------------------------------------------------------------------------------

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape(1, -1) # Convert 1D array to 2D array
    e_x = np.exp(x)
    return e_x / (np.sum(e_x, axis=1, keepdims=True) + 1e-16) # Add small value to avoid division by zero

def log_stable(x):
    """Compute log values for each sets of scores in x."""
    return np.log(x + np.exp(-16))

def uncertainty_measure(evidence):
    """
    Measure uncertainty based on the inverse of the variance of the evidence.
    Lower variance indicates higher uncertainty.
    """
    variance = np.var(evidence, axis=1, keepdims=True)
    # To avoid division by zero, add a small epsilon
    epsilon = 1e-6
    return 1.0 / (variance + 1)

#------------------------------------------------------------------------------------------
# ---- Functions for calculating likelihoods and salience for free energy calculations ----
#------------------------------------------------------------------------------------------

def calculate_kl_divergence(p, q):
    """Calculate KL divergence between two probability distributions."""
    return np.sum(p * log_stable(p / (q + np.exp(-16))))/16 #Divide by 16 to normalize the KL divergence

def calculate_shannon_entropy(p):
    """Calculate Shannon entropy of a probability distribution."""
    return -np.sum(p * log_stable(p))

def max_entropy_prior(distribution):
    """Calculate the max entropy prior for a given distribution."""
    entropy = np.sum([calculate_shannon_entropy(d) for d in distribution])
    return entropy

def get_likelihood(robot_id,observations,goals,agent_vars,prior,use_ep=False, consensus=False, alpha=0.9):
    """Calculate the likelihood of a goal being the target based on agent positions."""
    observed_poses_tensor = np.array([obs['position'] for obs in observations])
    goals_tensor = goals
    robot_types = agent_vars['agent_types'].copy()
    reward_configs = agent_vars['reward_configs']
    if not use_ep:
        prior = prior.flatten()
        likelihood = np.zeros(goals.shape[0],dtype=np.float64).flatten()
        robot_types = [obs['type'] for obs in observations]
        likelihood += calculate_joint_goal_probs(robot_id, observed_poses_tensor, goals_tensor, robot_types, reward_configs, agent_vars['max_distance_measure']).flatten()
    else:
        likelihood = np.tile(np.zeros(goals.shape[0],dtype=np.float64).flatten(),(len(observations),1))
        for predict_id, predict_type in enumerate(agent_vars['agent_types']):
            tmp_obs_type = ['none' if (predict_type == 'A' and robot_types[robot_id] == 'B') else robot_types[robot_id] for _ in range(len(robot_types))]
            tmp_obs_type[predict_id] = 's' if predict_id == robot_id else tmp_obs_type[predict_id]
            likelihood[predict_id] += calculate_joint_goal_probs(predict_id, observed_poses_tensor, goals_tensor, tmp_obs_type, reward_configs, agent_vars['max_distance_measure']).flatten()
    if consensus:
        prior = compute_consensus(prior, agent_vars['agent_types'])
        likelihood = compute_consensus(likelihood, agent_vars['agent_types'])
    
    posterior = np.array([likelihood[i] * prior[i] for i in range(len(prior))])

    return softmax(posterior)

def compute_consensus(likelihoods, agent_types):
    """Compute consensus likelihoods based on agent types."""
    consensus_likelihoods = np.zeros(likelihoods[0].shape)
    weights = np.array([2.0 if agent_type == 'B' else 0.5 for agent_type in agent_types])
    for idx in range(likelihoods.shape[1]):
        consensus_likelihoods[idx] = np.sum(likelihoods[:, idx] * weights)
    return softmax(consensus_likelihoods)

def custom_cdist(agent_id, x1, x2, types, max_distance=30.0):
    """
    Compute the pairwise distance between rows of x1 and rows of x2 based on measurement types.

    Args:
        x1 (np.ndarray): An array of shape (m, d)
        x2 (np.ndarray): An array of shape (n, d)
        types (list): A list of measurement types for each pair of rows.
    Returns:
        np.ndarray: An array of shape (m, n) with the pairwise distances.
    """
    assert len(types) == x1.shape[0], "Length of types must match number of rows in x1"

    evidence = np.zeros((x1.shape[0], x2.shape[0]), dtype=np.float64)
    goal_vectors = x2 - x1[agent_id, :]
    goal_azimuths = np.arctan2(goal_vectors[:, 1], goal_vectors[:, 0])
    
    for i, t in enumerate(types):
        if t == 's':  # Self-observation
            depth_multiplier = 1#min(max(2.0, 2.0 / np.min(np.abs(x1[i] - x2))), 100.0)
            value = depth_multiplier * np.exp(-np.linalg.norm(x1[i] - x2, axis=-1) / max_distance)
        elif t == 'A':
            value = np.exp(-np.linalg.norm(x1[i] - x2, axis=-1) / max_distance)
        elif t == 'B':
            observed_azimuth = np.arctan2(x1[i, 1] - x1[agent_id, 1], x1[i, 0] - x1[agent_id, 0])
            relative_azimuths = np.abs((goal_azimuths - observed_azimuth + np.pi) % (2 * np.pi) - np.pi)
            # Using cosine to value alignment, where 1 means perfectly aligned and -1 means opposite
            alignment = np.cos(relative_azimuths)
            value = alignment  # Exponential scaling for smoother transition
        else:
            value = np.zeros(x2.shape[0], dtype=np.float64) + 1.0 / x2.shape[0]
        
        evidence[i] = value
    uncertainty = uncertainty_measure(evidence)

    # Dynamically adjust the exploration factor based on uncertainty
    # exploration_factor = 0.01 * uncertainty

    # Introduce exploration based on uncertainty
    # evidence += exploration_factor * np.random.rand(*evidence.shape)

    if np.isnan(softmax(evidence)).any():
        print("NAN in distances", evidence, x1, x2, types, max_distance)
        s = 1
    return evidence

def calculate_joint_goal_probs(agent_id, agent_poses, goals, predict_types, reward_configs, max_distance=30.0):
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
    evidence = custom_cdist(agent_id, agent_poses, goals, predict_types, max_distance)

    # Convert distances to probabilities using softmax
    probabilities = softmax(evidence) # Apply softmax along the goal dimension

    # Initialize joint probabilities as an array of ones with the appropriate shape
    joint_probabilities = np.ones([num_goals] * num_agents, dtype=float)

    # Calculate joint probabilities
    for i in range(num_agents):
        joint_probabilities *= probabilities[i].reshape([num_goals if j == i else 1 for j in range(num_agents)]) 

    # Only return the specified configurations
    likelihood = np.array([joint_probabilities[tuple(config)] for config in reward_configs], dtype=np.float64)

    # Normalize the joint probabilities
    likelihood /= likelihood.sum()

    return likelihood


#------------------------------------------------------------------------------------------
# ----------------- Functions for simulating observations and agent decision-making -------
#------------------------------------------------------------------------------------------

def simulate_observation(true_position, observation_error_std=2.0, sim_type='A', observed_agent= None):
    """Simulate noisy observation of another agent's position."""
    observation = {'position': np.zeros((2,))-1e6, 'heading': float(-1e6), 'type': sim_type, 'observed_agent': observed_agent}
    error_std = np.sqrt(observation_error_std)
    if sim_type in ['s']: # Assume no noise for self-observation or simulated observation
        observation['position'] = true_position[:2]
        observation['heading'] = true_position[2]       
    else:
        observation['position'] = true_position[:2] + np.random.normal(0, error_std, true_position[:2].shape)
        observation['heading'] = true_position[2] + np.random.normal(0, error_std*0.1) 

    return observation

def predict_agent_position(agent_position, velocity, heading):
    """Predict agent's next position based on chosen velocity and heading."""
    agent_prediction = np.copy(agent_position)
    agent_prediction[0] += velocity * np.cos(heading)
    agent_prediction[1] += velocity * np.sin(heading)
    agent_prediction[2] = heading
    return agent_prediction

def parse_observations(observations):
    """Parse observations to get agent positions."""
    agent_positions = np.zeros((len(observations), len(observations[0]['position']) + 1))
    for idx, observation in enumerate(copy.deepcopy(observations)):
        agent_positions[idx] = np.concatenate((observation['position'], [observation['heading']]),axis=0)
    return agent_positions.copy()

#------------------------------------------------------------------------------------------
# ----------------- Main function for agent decision-making -------------------------------
#------------------------------------------------------------------------------------------

def make_decision(agent_vars, use_ep=True):
    """Agent decision-making based on active inference to encourage convergence on a shared goal."""
    #Parse arguments
    agent_id = agent_vars['agent_id']
    agent_type = agent_vars['agent_type']
    agent_positions = agent_vars['agent_positions']
    prior = agent_vars['prior']
    observation_error_std = agent_vars['observation_error_std']
    use_threading = agent_vars['use_threading']

    # Initialize variables for decision-making
    observations = []
    # Start decision loops for each possible action
    for idx in range(agent_positions.shape[0]):
        if idx == agent_id:
            observations.append(simulate_observation(agent_positions[idx], 0, 's',idx)) # No noise for self-observation
        else:
            observations.append(simulate_observation(agent_positions[idx], observation_error_std, agent_type,idx))

    observed_positions = parse_observations(observations)
    
    # Calculate the likelihood of each goal being the target based on agent measurements -> May be able to turn into NN
    consensus_prior = prior
    if use_ep:
        consensus_prior = softmax(compute_consensus(prior, agent_vars['agent_types']))
    
    # Calculate the best policy
    if agent_vars['use_mcts'] and not use_threading:
        best_node = mcts_decision(observed_positions, consensus_prior, observations, agent_vars)
        best_action = (best_node.velocity, best_node.heading)
        best_value = best_node.value
    elif agent_vars['use_mcts'] and use_threading:
        best_node = mcts_decision_threaded(observed_positions, consensus_prior, observations, agent_vars)
        best_action = (best_node.velocity, best_node.heading)
        best_value = best_node.value
    else:
        best_velocity, best_heading, best_value = choice_heuristic(observed_positions, observations, consensus_prior, agent_vars, use_ep=use_ep, consensus=use_ep)
        best_action = (best_velocity, best_heading)

    return best_action, observations, best_value

#------------------------------------------------------------------------------------------
# ----------------- Functions for parsing arguments and running the simulation ------------
#------------------------------------------------------------------------------------------

def parse_args_by_agent(args):
    """Parse arguments by agent."""
    agent_vars = []
    for agent_id in range(len(args['agent_positions'])):
        agent_dict = {
            'agent_id': agent_id,
            'agent_type': args['agent_types'][agent_id],
            'agent_positions': args['agent_positions'],
            'goals': np.array(args['goals']),
            'velocity_options': args['velocity_options'],
            'heading_options': args['heading_options'],
            'num_actions': args['num_actions'],
            'max_distance_measure': args['max_distance_measure'],
            'observation_error_std': args['observation_error_std'],
            'prior': args['prior'],
            'use_ep': args['use_ep'],
            'use_mcts': args['use_mcts'],
            'agent_types': args['agent_types'],
            'reward_configs': args['reward_configs'],
            'mcts_iterations': args['mcts_iterations'],
            'horizon': args['horizon'],
            'use_threading': args['use_threading']
        }                                                  
        if args['use_ep']:
            agent_dict['prior'] = np.tile(args['prior'], (len(args['agent_positions']), 1))      
        agent_vars.append(agent_dict)
    return agent_vars

def check_convergence(positions, goals, convergence_type, max_distance=1.0):
    """Check if agents have converged to the same goal."""
    distances_to_goals = [np.linalg.norm(goals - pos[:2], axis=1) for pos in positions]
    distances_to_selected_goal = [np.min(distances) for distances in distances_to_goals]
    selected_goals = [np.argmin(distances) for distances in distances_to_goals]
    if convergence_type == 'converge':
        all_same_goal = [selected_goals[0] == which_goal for which_goal in selected_goals]
        check = (np.array(distances_to_selected_goal)<max_distance).all() and all(all_same_goal)
        return check, selected_goals
    elif convergence_type == 'exclusive':
        all_different_goals = len(selected_goals) == len(set(selected_goals))
        check = (np.array(distances_to_selected_goal)<max_distance).all() and all_different_goals
        return check, selected_goals

def run_simulation(args, max_iterations=100):
    """Run the simulation until both agents converge to the same goal or max iterations reached."""
    return_args = {}
    agent_vars = parse_args_by_agent(args)
    num_agents = len(args['agent_positions'])
    current_positions = np.copy(args['agent_positions'])
    true_positions = np.copy(current_positions)
    goals = args['goals']
    return_args['positions'] = [np.copy(current_positions)]
    free_energy_scores = []
    use_ep = args['use_ep']
    converged_count = 0

    for iteration in range(max_iterations):
        # Make decisions for all agents
        start_time = time.time()
        decisions = []
        observations = []
        if iteration >= 20:
            a = 1
            

        for idx in range(num_agents):
            decision, observation, free_energy_score = make_decision(agent_vars[idx], use_ep) # Make decision for agent
            decisions.append(decision) # Save decisions for all agents
            observations.append(observation) # Save observations for all agents
            free_energy_scores.append((free_energy_score, idx, iteration)) # Save free energy scores for all agents
        # Update agent positions based on their decisions
        for agent_id, (velocity, heading) in enumerate(decisions):
            dx = velocity * np.cos(heading)
            dy = velocity * np.sin(heading)
            true_positions[agent_id] += np.array([dx, dy, 0])
            true_positions[agent_id][2] = heading
        # decided_velocities = [decision[0] for decision in decisions]
        for agent_id in range(num_agents):
            agent_vars[agent_id]['agent_positions'] = np.copy(true_positions)
            agent_prior = agent_vars[agent_id]['prior'].copy()
            agent_vars[agent_id]['prior'] = get_likelihood(agent_id,observations[agent_id],
                                                            goals, agent_vars[agent_id], agent_prior, use_ep)
                    
        # Check if agents have converged to the same goal
        current_positions = np.copy(true_positions)
        convergence_check, selected_goal = check_convergence(current_positions, goals, args['convergence_type'], args['max_distance_measure']/args['env_size'])
        # Get execution time and print progress
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\rIteration {iteration+1}: Agents have selected goals {selected_goal}. Execution Time: {execution_time}s", end=' ')
        # Save current positions for plotting
        return_args['positions'].append(np.copy(current_positions))
        if convergence_check:
            converged_count += 1
            if converged_count >= 5:
                print(f"Agents have converged to Goal {selected_goal[0]} after {iteration + 1} iterations. Use EP: {use_ep}")
                return {'positions': current_positions, 'converged': True, 'iteration': iteration, 
                        'plot_args': return_args, 'priors': [agent_vars[i]['prior'] for i in range(num_agents)], 'energy_scores': free_energy_scores}

    print("Agents did not converge to the same goal within the maximum iterations.")
    return {'positions': current_positions, 'converged': False, 'iteration': iteration, 
            'plot_args': return_args, 'priors': [agent_vars[i]['prior'] for i in range(num_agents)], 'energy_scores': free_energy_scores}

#------------------------------------------------------------------------------------------
# ----------------- Functions for MCTS ----------------------------------------------------
#------------------------------------------------------------------------------------------

def choice_heuristic(current_positions, observations, prior, agent_params, use_ep=False, consensus=False):
    """Find velocity and heading that minimizes the free energy for the agent."""
    velocity_options = agent_params['velocity_options']
    heading_options = agent_params['heading_options']
    goals = agent_params['goals']
    agent_id = agent_params['agent_id']
    
    num_velocity_options = len(velocity_options)
    num_heading_options = len(heading_options)
    
    predicted_positions = np.tile(current_positions, (agent_params['num_actions'], 1, 1))
    predicted_observations = [copy.deepcopy(observations) for _ in range(agent_params['num_actions'])]

    velocities = np.repeat(velocity_options, num_heading_options)
    headings = np.tile(heading_options, num_velocity_options)
    
    dx = velocities * np.cos(headings)
    dy = velocities * np.sin(headings)
    
    agent_positions = predicted_positions[:, agent_id]
    agent_positions[:, 0] += dx
    agent_positions[:, 1] += dy
    agent_positions[:, 2] = headings

    for i, pos in enumerate(agent_positions):
        predicted_observations[i][agent_id]['position'] = pos[:2]
        predicted_observations[i][agent_id]['heading'] = pos[2]
    
    posteriors = [get_likelihood(agent_id, obs, goals, agent_params, prior, use_ep, consensus) for obs in predicted_observations]
    kl_divergences = [calculate_kl_divergence(prior, post) for post in posteriors]
    entropies = [max_entropy_prior(post) for post in posteriors]
    free_energies = np.array(entropies) + np.array(kl_divergences)
    
    best_idx = np.argmin(free_energies)
    best_velocity = velocities[best_idx]
    best_heading = headings[best_idx]
    best_score = free_energies[best_idx]
    
    return best_velocity, best_heading, best_score
           
def mcts_decision(positions, prior, observations, agent_params, horizon=5, iterations=1000):
    """Make a decision based on Monte Carlo Tree Search."""
    goals = agent_params['goals']
    num_actions = len(agent_params['velocity_options']) * len(agent_params['heading_options'])
    # Initialize MCTS tree
    root = MCTSNode(positions, prior, num_actions, observations)
    # Run MCTS for a set number of iterations
    for _ in range(iterations):
        node = selection(root)
        if not node.is_fully_expanded():
            expansion(node, agent_params, goals)
        child_node = node.least_visited_child()
        reward = simulation(child_node, goals, agent_params)
        backpropagation(child_node, reward)

    best_action_node = root.best_child(exploration_weight=0)
   
    return best_action_node

def simulate_and_backpropagate(node, goals, agent_params):
    """Simulate the agent's actions and backpropagate the reward up the tree."""
    reward = simulation(node, goals, agent_params)
    backpropagation(node, reward)
    return reward

def mcts_decision_threaded(positions, prior, observations, agent_params):
    """Make a decision based on Monte Carlo Tree Search using multi-threading."""
    goals = agent_params['goals']
    horizon = agent_params['horizon']
    iterations = agent_params['mcts_iterations']
    num_actions = len(agent_params['velocity_options']) * len(agent_params['heading_options'])
    # Initialize MCTS tree
    root = MCTSNode(positions, prior, num_actions, observations)

    # Determine the number of workers based on CPU cores
    num_cores = multiprocessing.cpu_count()

    start_time = time.perf_counter()  # Start timing
    # Create a ProcessPoolExecutor with the number of workers equal to the number of CPU cores
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for _ in range(iterations):
            node = selection(root)
            if not node.is_fully_expanded():
                expansion(node, agent_params, goals)
            child_node = node.least_visited_child()
            if child_node is None:
                continue
            futures.append(executor.submit(simulate_and_backpropagate, child_node, goals, agent_params))

        # Collect results
        for future in as_completed(futures):
            reward = future.result()  # This will wait for the result and backpropagate the reward

    end_time = time.perf_counter()  # End timing
    execution_time = end_time - start_time
    print(f"MCTS execution time: {execution_time} seconds")

    best_action_node = root.best_child(exploration_weight=0)
    return best_action_node

def selection(node):
    """Select the best child node based on UCB1."""
    while node.is_fully_expanded():
        node = node.best_child()
    return node

def expansion(node, agent_params, goals):
    """Expand the tree by adding new child nodes."""
    velocity_options = agent_params['velocity_options']
    heading_options = agent_params['heading_options']
    max_distance_measure = agent_params['max_distance_measure']
    agent_id = agent_params['agent_id']
    use_ep = agent_params['use_ep']
    current_positions = np.copy(node.position)
    num_actions = agent_params['num_actions']
    node_prior = np.copy(node.prior)

    for velocity in velocity_options:
        for heading in heading_options:
            predicted_position = predict_agent_position(current_positions[agent_id], velocity, heading)
            new_agent_positions = np.copy(current_positions)
            new_agent_positions[agent_id] = predicted_position
            predicted_observations = copy.deepcopy(node.observations)
            predicted_observations[agent_id]['position'] = predicted_position[:2]
            predicted_observations[agent_id]['heading'] = predicted_position[2]
            #Predict how what you would do would differ from the prior (what the system is doing - complexity)
            new_prior = get_likelihood(agent_id, predicted_observations, goals, agent_params, node_prior, use_ep, consensus=use_ep)
            child_node = MCTSNode(new_agent_positions, new_prior, num_actions, predicted_observations, velocity, heading, parent=node)
            node.add_child(child_node)

def simulation(node, goals, agent_params, horizon):
    """Simulate the agent's actions and calculate the free energy."""
    agent_id = agent_params['agent_id']
    use_ep = agent_params['use_ep']
    current_prior = np.copy(node.prior)
    current_positions = np.copy(node.position)
    predicted_observations = copy.deepcopy(node.observations)
    total_free_energy = np.inf

    for _ in range(horizon):
        velocity, heading, posterior = choice_heuristic(current_positions, predicted_observations, current_prior, agent_params, use_ep, consensus=use_ep)
        current_positions[agent_id] = predict_agent_position(current_positions[agent_id], velocity, heading)       
        predicted_observations[agent_id]['position'] = np.copy(current_positions[agent_id][:2])
        predicted_observations[agent_id]['heading'] = current_positions[agent_id][2]
        kl_divergence = calculate_kl_divergence(current_prior, posterior)
        entropy = calculate_shannon_entropy(posterior)
        total_free_energy = np.min((total_free_energy, entropy + kl_divergence))
        current_prior = posterior

    return total_free_energy  # Negative because MCTS maximizes reward

def backpropagation(node, reward):
    """Backpropagate the reward up the tree."""
    while node is not None:
        node.update(reward)
        node = node.parent

#------------------------------------------------------------------------------------------
# ----------------- Functions for plotting and parsing free energy scores -----------------
#------------------------------------------------------------------------------------------

class PlotSim:
    """Class to plot the simulation of agents moving towards a goal."""
    def __init__(self, num_agents, goals, env_size=15, padding = 5):
        # Initialize plotting objects
        self.fig, self.ax = plt.subplots()
        plt.xlim(-5, env_size+padding)
        plt.ylim(-5, env_size+padding)
        self.cmap = Set1_9.mpl_colors
        self.agent_paths = [self.ax.plot([], [], 'o-', markersize=3, linewidth=1, alpha=0.5, color=self.cmap[i])[0] for i in range(num_agents)]
        self.goal_plots = [self.ax.plot(goal[0], goal[1], 'x', markersize=10, color='purple')[0] for goal in goals]  # Plot goals
        self.all_rosbots = []  # Dictionary to store all rosbot elements

    def init(self):
        """Initialize the background of the plot."""
        agent_id = 0
        for agent_path in self.agent_paths:
            agent_path.set_data([], [])
            self.all_rosbots.append(self.init_robot(self.cmap[agent_id]))
            agent_id += 1
        return self.agent_paths


    def update(self, frame, args):
        """Update the plot data for each agent."""
        agent_positions = args['positions'][frame]
        
        # Update plot data for each agent
        for agent_id, agent_path in enumerate(self.agent_paths):
            xnew, ynew, heading = agent_positions[agent_id]
            self.update_robot(self.all_rosbots[agent_id], (xnew,ynew), heading, color=self.cmap[agent_id])
            xdata, ydata = agent_path.get_data()
            xdata = np.append(xdata, xnew)
            ydata = np.append(ydata, ynew)
            agent_path.set_data(xdata, ydata)
        #Update title
        self.ax.set_title(f'Iteration {frame}')
        return self.agent_paths

    def init_robot(self, color='r'):
        """Initialize parts of the Robot with placeholders for their geometric data"""
        elements = {
            'red_box': self.ax.fill([], [], color=color, alpha=0.3)[0],
            'left_tires': self.ax.fill([], [], color='k')[0],
            'right_tires': self.ax.fill([], [], color='k')[0],
            'light_bar': self.ax.fill([], [], color='k', alpha=0.5)[0],
        }
        return elements

    def update_robot(self, elements, position, heading_ang, color, scale=0.3):
        """Update the position and orientation of the Robot parts"""
        rosbot_scale = scale

        # Define the parts with their unscaled coordinates
        parts = {
            'red_box': np.array([[-2, -2, 2, 2], [1.6, -1.6, -1.6, 1.6]]),
            'left_tires': np.array([
                [0.35, 0.35, 2, 2, 0.35, -0.35, -2, -2, -0.35, -0.35],
                [1.6, 1.9, 1.9, 1.6, 1.6, 1.6, 1.6, 1.9, 1.9, 1.6]
            ]),
            'right_tires': np.array([
                [0.35, 0.35, 2, 2, 0.35, -0.35, -2, -2, -0.35, -0.35],
                [-1.6, -1.9, -1.9, -1.6, -1.6, -1.6, -1.6, -1.9, -1.9, -1.6]
            ]),
            'light_bar': np.array([[-0.6, -0.6, -1.25, -1.25], [1.2, -1.2, -1.2, 1.2]]),
        }

        # Apply scaling, rotation, and translation
        rotation = np.array([
            [np.cos(heading_ang), -np.sin(heading_ang)],
            [np.sin(heading_ang), np.cos(heading_ang)]
        ])
        for name, part in parts.items():
            scaled_part = rosbot_scale * part
            transformed = rotation @ scaled_part + np.array(position)[:, None]
            elements[name].set_xy(transformed.T)
            if name in ['red_box']:
                elements[name].set_facecolor(color)  # Apply color only to these parts

def parse_free_energy_scores(free_energy_scores, num_iterations):
    """Parse the free energy scores for each agent over time."""
    res_list = []; ending_energy = []
    for entry in free_energy_scores:
        avg_scores = entry[0] #np.min([score[0] for score in entry[0]])
        agent = entry[1]
        iteration = entry[2]
        res_list.append([avg_scores, agent, iteration])
        if iteration == num_iterations:
            ending_energy.append(avg_scores)
    return res_list, ending_energy

def plot_energy(data, num_agents):
    """Plot the free energy scores for each agent over time."""
    cmap = Set1_9.mpl_colors
    fig, ax = plt.subplots()
    df = pd.DataFrame(data, columns=['Avg Free Energy', 'Agent', 'Iteration'])
    # Plot the free energy scores for each agent
    for i in range(num_agents):
        agent_data = df[df['Agent'] == i]
        ax.plot(agent_data['Iteration'], agent_data['Avg Free Energy'], label=f'Agent {i}', color=cmap[i])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Free Energy')
    ax.set_title('Free Energy Scores for Each Agent')
    ax.legend()
    # Show plot
    plt.show()
# TODO: Fix this for threading
# if __name__ == '__main__':
#     # Get arguments from the user
#     args=json.loads(sys.argv[1])
#     results = run_simulation(args)
#     # Store results in a JSON file
#     with open('results.json', 'w') as f:
#         json.dump(results, f)
#     print("Simulation complete.")