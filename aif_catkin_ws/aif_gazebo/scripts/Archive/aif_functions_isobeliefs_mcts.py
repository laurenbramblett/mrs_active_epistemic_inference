import numpy as np
from matplotlib.patches import Polygon, Circle
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set1_9
import copy, time, json, sys
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

def get_likelihood(robot_id,observations,goals,system_dict,prior,max_distance_measure=1.0,use_ep=False, consensus=False, alpha=0.9):
    """Calculate the likelihood of a goal being the target based on agent positions."""
    robot_pose = observations[robot_id]['position']
    if not use_ep:
        likelihood = np.zeros(goals.shape[0]).flatten()
        for observation in observations:
            tmp_obs_type = observation['type']
            likelihood += salience(robot_pose,observation, tmp_obs_type, goals, max_distance_measure).flatten()
    else:
        likelihood = np.tile(np.zeros(goals.shape[0]).flatten(),(len(observations),1))
        robot_type = system_dict['agent_types'][robot_id]
        for predict_id, predict_type in enumerate(system_dict['agent_types']):
            for obs_id,observation in enumerate(observations):
                tmp_obs_type = ('self' if predict_id == obs_id else predict_type)
                tmp_obs_type = ('none' if (robot_type in ['B'] and predict_type in ['A']) else tmp_obs_type)
                likelihood[predict_id] += salience(observations[predict_id]['position'], observation, tmp_obs_type,goals,max_distance_measure).flatten()
        if consensus:
            likelihood = compute_consensus(likelihood, system_dict['agent_types'])

    # TODO: Incorporate the prior into the likelihood
    posterior = softmax(likelihood)
    
    return posterior

def compute_consensus(likelihoods, agent_types):
    """Compute consensus likelihoods based on agent types."""
    consensus_likelihoods = np.zeros(likelihoods[0].shape)
    weights = np.array([2.0 if agent_type == 'B' else 0.5 for agent_type in agent_types])
    for idx in range(likelihoods.shape[1]):
        consensus_likelihoods[idx] = np.sum(likelihoods[:, idx] * weights)
    return consensus_likelihoods

def salience(robot_pose, observation, predict_type, goals, max_distance_measure=1.0, use_ep=False):
    """Calculate the salience of each goal based on depth and azimuth measurements from the robot."""
    goal_vectors = goals - robot_pose[:2]
    goal_azimuths = np.arctan2(goal_vectors[:, 1], goal_vectors[:, 0])
    saliences = np.zeros(len(goals))

    if predict_type in ['A','self']:
        # For depth, compute how close each goal is to the observed depth
        depth_differences = np.linalg.norm(observation['position'] - goals, axis=1)
        if predict_type == 'self':
            #As robot gets closer -- becomes more salient
            depth_multiplier = max(2.0, 2.0/min(depth_differences))
            depth_salience = depth_multiplier*np.exp(-depth_differences / max_distance_measure)  # exponential decay based on depth difference
        else:
            depth_salience = np.exp(-depth_differences / max_distance_measure)  # exponential decay based on depth difference
        saliences += depth_salience
    
    if predict_type in ['B']:
        # For azimuth, compute alignment of each goal with the observed azimuth
        observed_azimuth = np.arctan2(observation['position'][1]-robot_pose[1], observation['position'][0]-robot_pose[0])
        relative_azimuths = np.abs((goal_azimuths - observed_azimuth + np.pi) % (2 * np.pi) - np.pi)
        azimuth_salience = 1./8 * np.exp(- relative_azimuths / np.pi)  # normalize and invert to make smaller angles more salient
        saliences += azimuth_salience
        # Compute if observed robot is heading towards the goal
        # heading_to_goal = (np.arctan2(goals[:, 1] - observation['position'][1], goals[:, 0] - observation['position'][0]) - observation['heading'] + np.pi) % (2 * np.pi) - np.pi
        # heading_salience = 1./8 * np.exp(- np.abs(heading_to_goal) / np.pi)
        # saliences += heading_salience            

    return saliences

#------------------------------------------------------------------------------------------
# ----------------- Functions for simulating observations and agent decision-making -------
#------------------------------------------------------------------------------------------

def simulate_observation(true_position, observation_error_std=2.0, sim_type='A', observed_agent= None):
    """Simulate noisy observation of another agent's position."""
    observation = {'position': np.zeros((2,))-1e6, 'heading': float(-1e6), 'type': sim_type, 'observed_agent': observed_agent}
    error_std = np.sqrt(observation_error_std)
    if sim_type in ['self', 'sim']: # Assume no noise for self-observation or simulated observation
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

def make_decision(args, system_dict, use_ep=True):
    """Agent decision-making based on active inference to encourage convergence on a shared goal."""
    #Parse arguments
    agent_id = args['agent_id']
    agent_type = args['agent_type']
    agent_positions = args['agent_positions']
    prior = args['prior']
    observation_error_std = args['observation_error_std']
    horizon = system_dict['horizon']
    mcts_iterations = system_dict['mcts_iterations']
    use_threading = system_dict['use_threading']

    # Initialize variables for decision-making
    observations = []
    print(f"Agent Positions: {agent_positions}")
    # Start decision loops for each possible action
    for idx in range(agent_positions.shape[0]):
        if idx == agent_id:
            observations.append(simulate_observation(agent_positions[idx], 0, 'self',idx))
        else:
            observations.append(simulate_observation(agent_positions[idx], observation_error_std, agent_type,idx))

    observed_positions = parse_observations(observations)
    
    # Calculate the likelihood of each goal being the target based on agent measurements -> May be able to turn into NN
    consensus_prior = prior
    if use_ep:
        consensus_prior = softmax(compute_consensus(prior, system_dict['agent_types']))
    
    # Calculate the best policy
    if system_dict['use_mcts'] and not use_threading:
        best_node = mcts_decision(observed_positions, consensus_prior, observations, args, system_dict, horizon, mcts_iterations)
        best_action = (best_node.velocity, best_node.heading)
        best_value = best_node.value
    elif system_dict['use_mcts'] and use_threading:
        best_node = mcts_decision_threaded(observed_positions, consensus_prior, observations, args, system_dict, horizon, mcts_iterations)
        best_action = (best_node.velocity, best_node.heading)
        best_value = best_node.value
    else:
        best_velocity, best_heading, best_value = choice_heuristic(observed_positions, observations, consensus_prior, args, system_dict, use_ep=use_ep)
        best_action = (best_velocity, best_heading)

    return best_action, observations, best_value

#------------------------------------------------------------------------------------------
# ----------------- Functions for parsing arguments and running the simulation ------------
#------------------------------------------------------------------------------------------

def parse_args_by_agent(args):
    """Parse arguments by agent."""
    agent_vars = []
    for agent_id in range(len(args['agent_positions'])):
        agent_dict = {}
        agent_dict['agent_id'] = agent_id
        agent_dict['agent_type'] = args['agent_types'][agent_id]
        agent_dict['agent_positions'] = args['agent_positions']
        agent_dict['goals'] = args['goals']
        agent_dict['velocity_options'] = args['velocity_options']
        agent_dict['heading_options'] = args['heading_options']
        agent_dict['max_distance_measure'] = args['max_distance_measure']
        agent_dict['observation_error_std'] = args['observation_error_std']
        agent_dict['prior'] = args['prior']
        if args['use_ep']:
            agent_dict['prior'] = np.tile(args['prior'], (len(args['agent_positions']), 1))      
        agent_vars.append(agent_dict)
    return agent_vars

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

        for idx in range(num_agents):
            decision, observation, free_energy_score = make_decision(agent_vars[idx],args, use_ep) # Make decision for agent
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
            observations[agent_id][agent_id]['position'] = true_positions[agent_id][:2]
            observations[agent_id][agent_id]['heading'] = true_positions[agent_id][2]
            agent_prior = agent_vars[agent_id]['prior']
            agent_vars[agent_id]['prior'] = get_likelihood(agent_id,observations[agent_id],
                                                            goals, args, agent_prior, args['max_distance_measure'], use_ep)
                    
        # Check if agents have converged to the same goal
        current_positions = np.copy(true_positions)
        distances_to_goals = [np.linalg.norm(goals - pos[:2], axis=1) for pos in current_positions]
        distances_to_selected_goal = [np.min(distances) for distances in distances_to_goals]
        selected_goal = [np.argmin(distances) for distances in distances_to_goals]
        all_same_goal = [selected_goal[0] == which_goal for which_goal in selected_goal]
        # Get execution time and print progress
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\rIteration {iteration+1}: Agents have selected goals {selected_goal}. Execution Time: {execution_time}s", end=' ')
        # Save current positions for plotting
        return_args['positions'].append(np.copy(current_positions))
        if (np.array(distances_to_selected_goal)<args['max_distance_measure']/30).all() and all(all_same_goal):
            converged_count += 1
            if converged_count >= 15:
                print(f"Agents have converged to Goal {selected_goal[0]} after {iteration + 1} iterations.")
                return {'positions': current_positions, 'converged': True, 'iteration': iteration, 
                        'plot_args': return_args, 'priors': [agent_vars[i]['prior'] for i in range(num_agents)], 'energy_scores': free_energy_scores}

    print("Agents did not converge to the same goal within the maximum iterations.")
    return {'positions': current_positions, 'converged': False, 'iteration': iteration, 
            'plot_args': return_args, 'priors': [agent_vars[i]['prior'] for i in range(num_agents)], 'energy_scores': free_energy_scores}

#------------------------------------------------------------------------------------------
# ----------------- Functions for MCTS ----------------------------------------------------
#------------------------------------------------------------------------------------------

def choice_heuristic(current_positions, observations, prior, agent_params, system_dict, use_ep=False, consensus=False):
    """Find velocity and heading that minimizes the free energy for the agent."""
    velocity_options = agent_params['velocity_options']
    heading_options = agent_params['heading_options']
    max_distance_measure = agent_params['max_distance_measure']
    goals = agent_params['goals']
    agent_id = agent_params['agent_id']
    use_ep = system_dict['use_ep']
    current_prior = prior
    predicted_positions = np.copy(current_positions)
    predicted_observations = copy.deepcopy(observations)
    scores = []

    for velocity in velocity_options:
        for heading in heading_options:
            predicted_positions[agent_id] = predict_agent_position(current_positions[agent_id], velocity, heading)       
            predicted_observations[agent_id]['position'] = predicted_positions[agent_id][:2]
            predicted_observations[agent_id]['heading'] = predicted_positions[agent_id][2]
            posterior = get_likelihood(agent_id, predicted_observations, goals, system_dict, current_prior, max_distance_measure, use_ep, consensus=use_ep)
            kl_divergence = calculate_kl_divergence(current_prior, posterior)
            entropy = max_entropy_prior(posterior)
            total_free_energy = entropy + kl_divergence
            scores.append((total_free_energy, velocity, heading, posterior))
    
    best_score = min(scores, key=lambda x: x[0])
    return best_score[1], best_score[2], best_score[0]
            
def mcts_decision(positions, prior, observations, agent_params, system_dict, horizon=5, iterations=1000):
    """Make a decision based on Monte Carlo Tree Search."""
    goals = agent_params['goals']
    num_actions = len(agent_params['velocity_options']) * len(agent_params['heading_options'])
    # Initialize MCTS tree
    root = MCTSNode(positions, prior, num_actions, observations)
    # Run MCTS for a set number of iterations
    for _ in range(iterations):
        node = selection(root)
        if not node.is_fully_expanded():
            expansion(node, agent_params, system_dict, goals)
        child_node = node.least_visited_child()
        reward = simulation(child_node, goals, system_dict, agent_params, horizon)
        backpropagation(child_node, reward)

    best_action_node = root.best_child(exploration_weight=0)
   
    return best_action_node

def simulate_and_backpropagate(node, goals, system_dict, agent_params, horizon):
    """Simulate the agent's actions and backpropagate the reward up the tree."""
    reward = simulation(node, goals, system_dict, agent_params, horizon)
    backpropagation(node, reward)
    return reward

def mcts_decision_threaded(positions, prior, observations, agent_params, system_dict, horizon=5, iterations=1000):
    """Make a decision based on Monte Carlo Tree Search using multi-threading."""
    goals = agent_params['goals']
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
                expansion(node, agent_params, system_dict, goals)
            child_node = node.least_visited_child()
            if child_node is None:
                continue
            futures.append(executor.submit(simulate_and_backpropagate, child_node, goals, system_dict, agent_params, horizon))

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

def expansion(node, agent_params, system_dict, goals):
    """Expand the tree by adding new child nodes."""
    velocity_options = agent_params['velocity_options']
    heading_options = agent_params['heading_options']
    max_distance_measure = agent_params['max_distance_measure']
    agent_id = agent_params['agent_id']
    use_ep = system_dict['use_ep']
    current_positions = np.copy(node.position)
    num_actions = system_dict['num_actions']
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
            new_prior = get_likelihood(agent_id, predicted_observations, goals, system_dict, node_prior, max_distance_measure, use_ep, consensus=use_ep)
            child_node = MCTSNode(new_agent_positions, new_prior, num_actions, predicted_observations, velocity, heading, parent=node)
            node.add_child(child_node)

def simulation(node, goals, system_dict, agent_params, horizon):
    """Simulate the agent's actions and calculate the free energy."""
    agent_id = agent_params['agent_id']
    use_ep = system_dict['use_ep']
    current_prior = np.copy(node.prior)
    current_positions = np.copy(node.position)
    predicted_observations = copy.deepcopy(node.observations)
    total_free_energy = np.inf

    for _ in range(horizon):
        velocity, heading, posterior = choice_heuristic(current_positions, predicted_observations, current_prior, agent_params, system_dict, use_ep, consensus=use_ep)
        current_positions[agent_id] = predict_agent_position(current_positions[agent_id], velocity, heading)       
        predicted_observations[agent_id]['position'] = np.copy(current_positions[agent_id][:2])
        predicted_observations[agent_id]['heading'] = current_positions[agent_id][2]
        # posterior = get_likelihood(agent_id, predicted_observations, goals, system_dict, max_distance_measure, use_ep, consensus=use_ep)
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