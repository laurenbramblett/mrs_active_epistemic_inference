import numpy as np
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set1_9
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import copy, time, os, sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing, itertools

# Pull methods from simulation directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Move one folder level up
parent_dir = os.path.dirname(script_dir)
sys.path.append(os.path.abspath(os.path.join(parent_dir, 'simulation_methods')))

from mcts_class import MCTSNode
from receding_horizon_functions import run_rhc


#------------------------------------------------------------------------------------------
# ----------------- Functions for sofmax and stable log calculations ----------------------
#------------------------------------------------------------------------------------------

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape(1, -1) # Convert 1D array to 2D array
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / (np.sum(e_x, axis=1, keepdims=True) + 1e-16)

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

def wrapToPi(x):
    """Wrap angle to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi

def calculate_goal_reward(agent_id, q, x, goals, reward_configs):
    """Calculate the reward for a given goal based on the joint probability distribution."""
    # aggregate_post = np.sum(q, axis=0, keepdims=True)
    closest_goal = np.min(np.linalg.norm(goals - x[:2], axis=1))
    return closest_goal

def calculate_proximity_penalty(agent_id, observations, min_distance=0.2):
    """Calculate the proximity penalty for agents being too close to each other."""
    observed_poses = np.array([obs['position'] for obs in observations])
    distances = compute_distances(observed_poses, observed_poses)
    np.fill_diagonal(distances, np.inf)

    # Get distances for the specific agent
    agent_distances = distances[agent_id]
    
    # Calculate penalty for all distances below the min_distance
    penalties = 1 / (agent_distances[agent_distances < min_distance] + 1e-4)
    total_penalty = np.sum(penalties) if len(penalties) > 0 else 0
    
    print("Total proximity penalty: ", total_penalty)
    return total_penalty * 10e3


#------------------------------------------------------------------------------------------
# ---- Functions for calculating likelihoods and salience for free energy calculations ----
#------------------------------------------------------------------------------------------

def calculate_kl_divergence(q):
    """Calculate KL divergence between two probability distributions. In our case it is picking one valid configuration"""
    aggregate_post = np.sum(q, axis=0, keepdims=True)
    p = np.where(aggregate_post == np.max(aggregate_post), 1, 0)
    return np.sum(p * log_stable(p / (q + np.exp(-16)))) #Divide by 16 to normalize the KL divergence

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
        likelihood = np.zeros(len(agent_vars['reward_configs']),dtype=np.float64).flatten()
        robot_types = [obs['type'] for obs in observations]
        likelihood += calculate_joint_goal_probs(robot_id, observed_poses_tensor, goals_tensor, robot_types, reward_configs, agent_vars['max_distance_measure']).flatten()
    else:
        likelihood = np.zeros((agent_vars['num_agents'],len(agent_vars['reward_configs'])),dtype=np.float64)
        for predict_id, predict_type in enumerate(agent_vars['agent_types']):
            tmp_obs_type = ['none' if (predict_type == 'A' and robot_types[robot_id] == 'B') else robot_types[robot_id] for _ in range(len(robot_types))]
            tmp_obs_type[predict_id] = 's' if predict_id == robot_id else tmp_obs_type[predict_id]
            robot_types = tmp_obs_type
            likelihood[predict_id] += calculate_joint_goal_probs(predict_id, observed_poses_tensor, goals_tensor, robot_types, reward_configs, agent_vars['max_distance_measure']).flatten()
    if consensus:
        prior = compute_consensus(prior, robot_types)
        likelihood = compute_consensus(likelihood, robot_types)

    posterior = np.array([(0.2 * likelihood[i]) + (0.8 * prior[i]) for i in range(len(prior))])

    return softmax(posterior)

def compute_consensus(likelihoods, agent_types):
    """Compute consensus likelihoods based on agent types."""
    consensus_likelihoods = np.zeros(likelihoods[0].shape)
    weights = np.array([1.0 if agent_type == 's' else 1.0 for agent_type in agent_types])
    for idx in range(likelihoods.shape[1]):
        consensus_likelihoods[idx] = np.sum(likelihoods[:, idx] * weights)
    return softmax(consensus_likelihoods)

def compute_distances(x, goals):
    """Compute the distances between agents and goals."""
    diff = x[:, np.newaxis, :] - goals[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    return distances

def custom_cdist(agent_id, x, goals, types, eta=30.0):
    """Compute the pairwise distance between rows of x and rows of goals based on measurement types."""
    assert len(types) == x.shape[0], "Length of types must match number of rows in x1"

    # Compute the pairwise differences to goals
    diff_to_goals = goals[np.newaxis, :, :] - x[:, np.newaxis, :]
    
    # Compute the pairwise distances to goals
    distances_to_goals = np.linalg.norm(diff_to_goals, axis=2)
    evidence_to_goals = np.exp(-1/eta * distances_to_goals)
    
    # Compute the pairwise angles to goals
    angles_to_goal = np.arctan2(diff_to_goals[:, :, 1], diff_to_goals[:, :, 0])
    
    # Compute the pairwise differences between robots
    diff_to_robot = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    
    # Compute the pairwise angles between robots
    angles_to_robot = np.arctan2(diff_to_robot[:, :, 1], diff_to_robot[:, :, 0])
    
    # Compute the relative angles
    relative_angles = np.abs((angles_to_goal[np.newaxis, :, :] - angles_to_robot[:, :, np.newaxis] + np.pi) % (2 * np.pi) - np.pi)
    
    # Using cosine to value alignment, where 1 means perfectly aligned and -1 means opposite
    alignment = np.cos(relative_angles)

    evidence = np.zeros((len(types), len(goals)), dtype=float) + 1/len(goals)
    for i in range(len(types)):
        if types[i] in ['s', 'A']:
            evidence[i] = evidence_to_goals[i]
        elif types[i] in ['B']:
            evidence[i] = alignment[i][agent_id]

    return np.squeeze(evidence)

# @timeit
def calculate_joint_goal_probs(agent_id, agent_poses, goals, predict_types, reward_configs, max_distance=30.0):
    """Calculate the joint probabilities of goals based on agent positions."""
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

def predict_agent_position(agent_position, velocity, heading, dt):
    """Predict agent's next position based on chosen velocity and heading."""
    agent_prediction = np.copy(agent_position)
    heading = wrapToPi(agent_position[2] + heading)
    agent_prediction[0] += velocity * np.cos(heading) * dt
    agent_prediction[1] += velocity * np.sin(heading) * dt
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
    elif agent_vars['use_rhc']:
        best_controls, best_value = run_rhc(agent_positions, agent_vars)
        best_velocities = best_controls[agent_id][0]
        best_action = (np.linalg.norm(best_velocities), np.arctan2(best_velocities[1], best_velocities[0]))
    elif agent_vars['greedy']:
        # g = greedy_decision(observed_positions[:,:2], observations, agent_vars)
        obs_types = [obs['type'] for obs in observations]
        goal = greedy_decision(observed_positions[:,:2], agent_vars['goals'], obs_types, agent_vars).flatten()
        best_action, best_value = move_towards_goal(agent_id, observed_positions, goal, agent_vars)
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
            'num_agents': len(args['agent_positions']),
            'home_base': args['home_base'],
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
            'use_threading': args['use_threading'],
            'use_rhc': args['use_rhc'],
            'greedy': args['greedy'],
            'dt': args['dt'],
            'convergence_threshold': args['convergence_threshold']
        }                                                  
        if args['use_ep']:
            agent_dict['prior'] = np.tile(1/len(args['reward_configs']), (len(args['agent_positions']), len(args['reward_configs'])))      
        else:
            agent_dict['prior'] = np.ones(len(args['reward_configs']), dtype=float) / len(args['reward_configs'])
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
    return_args['positions'] = [np.copy(current_positions[:,:2])]
    return_args['headings'] = [current_positions[:,2]]
    return_args['completed_goals'] = [np.repeat(False, len(goals))]
    free_energy_scores = []
    use_ep = args['use_ep']
    converged_count = 0
    goals_completed = np.copy(return_args['completed_goals'][-1])   
    dt = args['dt']
    converged_distance = args['convergence_threshold']

    for iteration in range(max_iterations):
        # Make decisions for all agents
        start_time = time.time()
        decisions = []
        observations = []            
        goals_completed = np.copy(return_args['completed_goals'][0]) 
        for idx in range(num_agents):
            # agent_vars[idx]['goals'] = choose_subset_goals(goals, num_agents)
            decision, observation, free_energy_score = make_decision(agent_vars[idx], use_ep) # Make decision for agent
            decisions.append(decision) # Save decisions for all agents
            observations.append(observation) # Save observations for all agents
            free_energy_scores.append((free_energy_score, idx, iteration)) # Save free energy scores for all agents
        # Update agent positions based on their decisions
        for agent_id, (velocity, heading) in enumerate(decisions):
            new_heading = wrapToPi(true_positions[agent_id][2] + heading)
            dx = velocity * np.cos(new_heading) * dt
            dy = velocity * np.sin(new_heading) * dt
            true_positions[agent_id] += np.array([dx, dy, 0])
            true_positions[agent_id][2] = new_heading
        # decided_velocities = [decision[0] for decision in decisions]
        for agent_id in range(num_agents):
            agent_vars[agent_id]['agent_positions'] = np.copy(true_positions)
            agent_prior = agent_vars[agent_id]['prior'].copy()
            agent_vars[agent_id]['prior'] = get_likelihood(agent_id,observations[agent_id],
                                                            goals, agent_vars[agent_id], agent_prior, use_ep)
            
        # Check if agents have converged to the same goal
        _, goals_completed = delete_goals(goals, true_positions, goals_completed, args['max_distance_measure']/args['env_size'])
                    
        # Check if agents have converged to the same goal
        current_positions = np.copy(true_positions)
        convergence_check, selected_goal = check_convergence(current_positions, goals, args['convergence_type'], args['convergence_threshold'])
        # Get execution time and print progress
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\rIteration {iteration+1}: Agents have selected goals {selected_goal}. Execution Time: {execution_time}s", end=' ')
        # Save current positions for plotting
        return_args['positions'].append(np.copy(current_positions[:,:2]))
        return_args['headings'].append(current_positions[:,2])
        return_args['completed_goals'].append(np.copy(goals_completed))
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
# ----------------- Functions for Epistemic Allocation ------------------------------------
#------------------------------------------------------------------------------------------
def choose_subset_goals(goals, agent_positions, params):
    """Choose a subset of goals for each agent to consider."""
    agent_types = params['agent_types']
    agent_types[params['agent_id']] = 's'
    evidences = custom_cdist(params['agent_id'], agent_positions, goals, agent_types, params['max_distance_measure'])
    probabilities = softmax(evidences)
    goal_indices = np.argsort(probabilities)[-params['num_agents']:]
    return goals[goal_indices]

def identify_reward_configs(num_goals):
    """Identify reward configurations for the joint goal probabilities."""
    tuple_elements = [i for i in range(num_goals[0])]
    configurations = list(itertools.permutations(tuple_elements))
    reward_configs = configurations
    return reward_configs

def delete_goals(goals, agent_positions, threshold=1.0):
    """Delete goals that are too close to agents."""
    distances = compute_distances(agent_positions, goals)
    # Identify goals that are within the threshold distance from any agent
    close_goals_indices = np.unique(np.where(distances < threshold)[1])
    # Delete these goals from the list
    remaining_goals = np.delete(goals, close_goals_indices, axis=0)
    
    return remaining_goals

def generate_spread_out_goals(num_goals, env_size, min_distance):
    goals = []
    while len(goals) < num_goals:
        new_goal = np.random.random(2) * env_size
        if all(np.linalg.norm(new_goal - goal) >= min_distance for goal in goals):
            goals.append(new_goal)
    return np.array(goals)

#------------------------------------------------------------------------------------------
# ----------------- Functions for Greedy --------------------------------------------------
#------------------------------------------------------------------------------------------   

def greedy_decision(positions, goals, obs_types, agent_params, num_goals=1):
    if len(goals) == 0:
        return []
    agent_id = agent_params['agent_id']
    evidence = custom_cdist(agent_id, positions, goals, obs_types, agent_params['max_distance_measure'])
    agent_evidence = evidence[agent_id]
    goal_idx = np.argsort(agent_evidence)[-num_goals:]
    goal = goals[goal_idx,:]
    return goal

def move_towards_goal(agent_id, positions, goal, agent_params):
    """Move towards the goal based on the greedy decision."""
    if len(goal) == 0:
        return (0, 0), 0
    best_velocity = np.min((np.linalg.norm(goal - positions[agent_id,:2]), agent_params['velocity_options'][-1]))
    angle_to_goal = np.arctan2(goal[1] - positions[agent_id][1], goal[0] - positions[agent_id][0])
    angle_diff = wrapToPi(angle_to_goal - positions[agent_id][2])
    best_heading = np.max((np.min((angle_diff, agent_params['heading_options'][-1])), agent_params['heading_options'][0]))
    value = 0
    return (best_velocity, best_heading), value
        

#------------------------------------------------------------------------------------------
# ----------------- Functions for MCTS ----------------------------------------------------
#------------------------------------------------------------------------------------------

def choice_heuristic(current_positions, observations, prior, agent_params, use_ep=False, consensus=False):
    """Find velocity and heading that minimizes the free energy for the agent."""
    velocity_options = agent_params['velocity_options']
    heading_options = agent_params['heading_options']
    goals = agent_params['goals']
    agent_id = agent_params['agent_id']
    dt = agent_params['dt']
    
    num_velocity_options = len(velocity_options)
    num_heading_options = len(heading_options)
    
    predicted_positions = np.tile(current_positions, (agent_params['num_actions'], 1, 1))
    predicted_observations = [copy.deepcopy(observations) for _ in range(agent_params['num_actions'])]

    velocities = np.repeat(velocity_options, num_heading_options)
    headings = np.tile(heading_options, num_velocity_options)
    
    new_headings = wrapToPi(predicted_positions[:,agent_id][:,2] + headings)
    dx = velocities * np.cos(new_headings)
    dy = velocities * np.sin(new_headings)
    
    agent_positions = predicted_positions[:, agent_id]
    agent_positions[:, 0] += dx * dt
    agent_positions[:, 1] += dy * dt
    agent_positions[:, 2] = new_headings

    for i, pos in enumerate(agent_positions):
        predicted_observations[i][agent_id]['position'] = pos[:2]
        predicted_observations[i][agent_id]['heading'] = pos[2]
    
    posteriors = [get_likelihood(agent_id, obs, goals, agent_params, prior, use_ep, consensus) for obs in predicted_observations]
    kl_divergences = [calculate_kl_divergence(post) for post in posteriors]
    entropies = [max_entropy_prior(post) for post in posteriors]
    min_distance =  [calculate_goal_reward(agent_params['agent_id'], posteriors[i], agent_positions[i], goals, agent_params['reward_configs']) for i in range(len(posteriors))]
    proximity_cost = [calculate_proximity_penalty(agent_params['agent_id'], predicted_observations[i], agent_params['convergence_threshold']-0.1) for i in range(len(agent_positions))]
    free_energies = np.array(entropies) + np.array(kl_divergences) + 10e-4*np.array(min_distance) + proximity_cost #Min distance breaks ties

    
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
    dt = agent_params['dt']

    for velocity in velocity_options:
        for heading in heading_options:
            predicted_position = predict_agent_position(current_positions[agent_id], velocity, heading, dt)
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
    dt = agent_params['dt']

    for _ in range(horizon):
        velocity, heading, posterior = choice_heuristic(current_positions, predicted_observations, current_prior, agent_params, use_ep, consensus=use_ep)
        current_positions[agent_id] = predict_agent_position(current_positions[agent_id], velocity, heading, dt)       
        predicted_observations[agent_id]['position'] = np.copy(current_positions[agent_id][:2])
        predicted_observations[agent_id]['heading'] = current_positions[agent_id][2]
        kl_divergence = calculate_kl_divergence(posterior)
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

def delete_goals(goals, agent_positions, goals_completed, threshold=1.0):
    """Delete goals that are too close to agents."""
    check = False
    distances = compute_distances(agent_positions[:,:2], goals)  
    #Find which goals were completed
    threshold_met = np.min(distances, axis=0) < threshold
    close_goals_indices = np.where(~goals_completed * threshold_met)[0]
    # Delete these goals from the list
    goals_completed[close_goals_indices] = True

    if len(close_goals_indices) > 0:
        # print(f"Deleted goals {close_goals_indices} that were too close to agents.")
        check = True 
    return check, goals_completed

class PlotSim:
    """Class to plot the simulation of agents moving towards a goal."""
    def __init__(self, num_agents, goals, robot_types, completed_task_img, not_completed_task_img, env_size=15, padding=5, scale = 0.5):
        # Initialize plotting objects
        self.fig, self.ax = plt.subplots()
        plt.xlim(-padding, env_size + padding)
        plt.ylim(-padding, env_size + padding)
        self.scale = scale
        self.cmap = Set1_9.mpl_colors
        self.agent_paths = [self.ax.plot([], [], 'o-', markersize=3, linewidth=1, alpha=0.5, color=self.cmap[i])[0] for i in range(num_agents)]
        self.all_rosbots = []  # List to store all robots
        self.robot_types = robot_types
        self.initial_task_positions = goals  # Store initial task positions
        self.completed_tasks = [False] * len(goals)  # Initialize all tasks as not completed
        self.task_icons = []
        # Load task icons
        self.completed_task_img = OffsetImage(completed_task_img, zoom=scale * 0.1)
        self.not_completed_task_img = OffsetImage(not_completed_task_img, zoom=scale * 0.1)
        # Initialize task icons
        for pos in goals:
            task_icon = AnnotationBbox(self.not_completed_task_img, pos, frameon=False, zorder=1)
            self.ax.add_artist(task_icon)
            self.task_icons.append(task_icon)

    def init(self):
        """Initialize the background of the plot."""
        agent_id = 0
        for agent_path in self.agent_paths:
            agent_path.set_data([], [])
            self.all_rosbots.append(self.init_robot(self.cmap[agent_id], self.robot_types[agent_id]))
            agent_id += 1
        return self.agent_paths

    def update(self, frame, args):
        """Update the plot data for each agent."""
        agent_positions = args['positions'][frame]
        agent_headings = args['headings'][frame]
        task_status = args['completed_goals'][frame]

        # Update plot data for each agent
        for agent_id, agent_path in enumerate(self.agent_paths):
            xnew, ynew = agent_positions[agent_id][:2]
            heading = agent_headings[agent_id]
            robot_type = self.robot_types[agent_id]
            self.update_robot(self.all_rosbots[agent_id], (xnew, ynew), heading, color=self.cmap[agent_id], robot_type=robot_type)
            xdata, ydata = agent_path.get_data()
            xdata = np.append(xdata, xnew)
            ydata = np.append(ydata, ynew)
            agent_path.set_data(xdata, ydata)
                # Update task icons based on task status
        for task_id, completed in enumerate(task_status):
            if completed:
                self.task_icons[task_id].offsetbox = self.completed_task_img
            else:
                self.task_icons[task_id].offsetbox = self.not_completed_task_img
        
        # Update title
        self.ax.set_title(f'Iteration {frame}')
        return self.agent_paths

    def init_robot(self, color='r', robot_type='A'):
        """Initialize parts of the Robot with placeholders for their geometric data"""
        elements = {}

        if robot_type == 'A':  # Ground vehicle
            elements['red_box'] = self.ax.fill([], [], color=color, alpha=1,zorder=2)[0]
            elements['left_tires'] = self.ax.fill([], [], color='k',zorder=2)[0]
            elements['right_tires'] = self.ax.fill([], [], color='k',zorder=2)[0]
            elements['light_bar'] = self.ax.fill([], [], color='k', alpha=0.5,zorder=2)[0]
            elements['lidar'] = self.ax.scatter([], [], color='k', s=30, alpha=0.9,zorder=2)
        
        elif robot_type == 'B':  # Drone
            elements['body'] = self.ax.scatter([], [], color=color, s=30, alpha=1,zorder=3)
            elements['arms'] = [self.ax.plot([], [], color=color, linewidth=5, alpha=1,zorder=2)[0] for _ in range(2)]
            elements['rotors'] = self.ax.scatter([], [], color=[0.5,0.5,0.5], s=30, alpha=0.6,zorder=3)
        return elements

    def update_robot(self, elements, position, heading_ang, color, robot_type='A'):
        """Update the position and orientation of the Robot parts"""
        if robot_type == 'B':  # Update for drone
            wspan = self.scale * 2  # Width of the drone arms
            rot = heading_ang
            center = np.array(position)
            
            pt1 = center + wspan * np.array([np.cos(rot + np.pi / 4), np.sin(rot + np.pi / 4)])
            pt2 = center - wspan * np.array([np.cos(rot + np.pi / 4), np.sin(rot + np.pi / 4)])
            pt3 = center - wspan * np.array([np.cos(rot - np.pi / 4), np.sin(rot - np.pi / 4)])
            pt4 = center + wspan * np.array([np.cos(rot - np.pi / 4), np.sin(rot - np.pi / 4)])

            body = np.array([pt1, pt2, pt3, pt4])
            elements['arms'][0].set_data(body[0:2, 0], body[0:2, 1])
            elements['arms'][1].set_data(body[2:4, 0], body[2:4, 1])
            elements['arms'][0].set_linewidth(3 * self.scale)  # Scale the line width
            elements['arms'][1].set_linewidth(3 * self.scale)  # Scale the line width

            elements['rotors'].set_offsets(body)
            elements['rotors'].set_sizes([self.scale**3 * 100] * 4)  # Scale the scatter size
            elements['body'].set_offsets(center.reshape(1,-1))
            elements['body'].set_facecolor(color)
            elements['body'].set_sizes([60 * self.scale**3])  # Scale the scatter size


        else:  # Update for ground vehicle
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

            rotation = np.array([
                [np.cos(heading_ang), -np.sin(heading_ang)],
                [np.sin(heading_ang), np.cos(heading_ang)]
            ])

            for name, part in parts.items():
                scaled_part = self.scale * part
                transformed = rotation @ scaled_part + np.array(position)[:, None]
                elements[name].set_xy(transformed.T)
                if name == 'red_box':
                    elements[name].set_facecolor(color)  # Apply color only to the ground vehicle parts
            lidar_position = np.array(position) + rotation @ np.array([0.6*self.scale, 0])  # Position in front of the vehicle
            elements['lidar'].set_offsets(lidar_position.reshape(1, -1))  # Update LiDAR position
            elements['lidar'].set_sizes([self.scale**2 * 50])  # Scale the scatter size

class PlotSim_fast:
    """Class to plot the simulation of agents moving towards a goal."""
    def __init__(self, num_agents, goals, robot_types, env_size=15, padding = 5, scale = 0.5):
        # Initialize plotting objects
        self.fig, self.ax = plt.subplots()
        plt.xlim(-padding, env_size + padding)
        plt.ylim(-padding, env_size + padding)
        self.scale = scale
        self.cmap = Set1_9.mpl_colors
        self.agent_paths = [self.ax.plot([], [], 'o-', markersize=3, linewidth=1, alpha=0.5, color=self.cmap[i])[0] for i in range(num_agents)]
        self.goal_plots = [self.ax.plot(goal[0], goal[1], 'x', markersize=10, color='purple')[0] for goal in goals]  # Plot goals
        self.all_rosbots = []  # List to store all robots
        self.robot_types = robot_types
        self.initial_task_positions = goals  # Store initial task positions
        self.completed_tasks = [False] * len(goals)  # Initialize all tasks as not completed

    def init(self):
        """Initialize the background of the plot."""
        agent_id = 0
        for agent_path in self.agent_paths:
            agent_path.set_data([], [])
            self.all_rosbots.append(self.init_robot(self.cmap[agent_id], self.robot_types[agent_id]))
            agent_id += 1
        return self.agent_paths

    def update(self, frame, args):
        """Update the plot data for each agent."""
        agent_positions = args['positions'][frame]
        agent_headings = args['headings'][frame]
        task_status = args['completed_goals'][frame]

        # Update plot data for each agent
        for agent_id, agent_path in enumerate(self.agent_paths):
            xnew, ynew = agent_positions[agent_id][:2]
            heading = agent_headings[agent_id]
            robot_type = self.robot_types[agent_id]
            self.update_robot(self.all_rosbots[agent_id], (xnew, ynew), heading, color=self.cmap[agent_id], robot_type=robot_type)
            xdata, ydata = agent_path.get_data()
            xdata = np.append(xdata, xnew)
            ydata = np.append(ydata, ynew)
            agent_path.set_data(xdata, ydata)

        for task_id, completed in enumerate(task_status):
            if completed:
                self.goal_plots[task_id].set_color('gray')

        #Update title
        self.ax.set_title(f'Iteration {frame}')
        return self.agent_paths

    def init_robot(self, color='r', robot_type='A'):
        """Initialize parts of the Robot with placeholders for their geometric data"""
        elements = {}

        if robot_type == 'A':  # Ground vehicle
            elements['red_box'] = self.ax.fill([], [], color=color, alpha=1,zorder=2)[0]
            elements['left_tires'] = self.ax.fill([], [], color='k',zorder=2)[0]
            elements['right_tires'] = self.ax.fill([], [], color='k',zorder=2)[0]
            elements['light_bar'] = self.ax.fill([], [], color='k', alpha=0.5,zorder=2)[0]
            elements['lidar'] = self.ax.scatter([], [], color='k', s=30, alpha=0.9,zorder=2)
        
        elif robot_type == 'B':  # Drone
            elements['body'] = self.ax.scatter([], [], color=color, s=30, alpha=1,zorder=3)
            elements['arms'] = [self.ax.plot([], [], color=color, linewidth=5, alpha=1,zorder=2)[0] for _ in range(2)]
            elements['rotors'] = self.ax.scatter([], [], color=[0.5,0.5,0.5], s=30, alpha=0.6,zorder=3)
        return elements

    def update_robot(self, elements, position, heading_ang, color, robot_type='A'):
        """Update the position and orientation of the Robot parts"""
        if robot_type == 'B':  # Update for drone
            wspan = self.scale * 2  # Width of the drone arms
            rot = heading_ang
            center = np.array(position)
            
            pt1 = center + wspan * np.array([np.cos(rot + np.pi / 4), np.sin(rot + np.pi / 4)])
            pt2 = center - wspan * np.array([np.cos(rot + np.pi / 4), np.sin(rot + np.pi / 4)])
            pt3 = center - wspan * np.array([np.cos(rot - np.pi / 4), np.sin(rot - np.pi / 4)])
            pt4 = center + wspan * np.array([np.cos(rot - np.pi / 4), np.sin(rot - np.pi / 4)])

            body = np.array([pt1, pt2, pt3, pt4])
            elements['arms'][0].set_data(body[0:2, 0], body[0:2, 1])
            elements['arms'][1].set_data(body[2:4, 0], body[2:4, 1])
            elements['arms'][0].set_linewidth(3 * self.scale)  # Scale the line width
            elements['arms'][1].set_linewidth(3 * self.scale)  # Scale the line width

            elements['rotors'].set_offsets(body)
            elements['rotors'].set_sizes([self.scale**3 * 100] * 4)  # Scale the scatter size
            elements['body'].set_offsets(center.reshape(1,-1))
            elements['body'].set_facecolor(color)
            elements['body'].set_sizes([60 * self.scale**3])  # Scale the scatter size


        else:  # Update for ground vehicle
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

            rotation = np.array([
                [np.cos(heading_ang), -np.sin(heading_ang)],
                [np.sin(heading_ang), np.cos(heading_ang)]
            ])

            for name, part in parts.items():
                scaled_part = self.scale * part
                transformed = rotation @ scaled_part + np.array(position)[:, None]
                elements[name].set_xy(transformed.T)
                if name == 'red_box':
                    elements[name].set_facecolor(color)  # Apply color only to the ground vehicle parts
            lidar_position = np.array(position) + rotation @ np.array([0.6*self.scale, 0])  # Position in front of the vehicle
            elements['lidar'].set_offsets(lidar_position.reshape(1, -1))  # Update LiDAR position
            elements['lidar'].set_sizes([self.scale**2 * 50])  # Scale the scatter size

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