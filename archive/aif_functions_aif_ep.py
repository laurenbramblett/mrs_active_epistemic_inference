import numpy as np
from matplotlib.patches import Polygon, Circle
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set1_9
import copy
import pandas as pd
import itertools

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum()

def log_stable(x):
    """Compute log values for each sets of scores in x."""
    return np.log(x + np.exp(-16))

def simulate_observation(true_position, observation_error_std=2.0, sim_type='A', observed_agent= None):
    """Simulate noisy observation of another agent's position."""
    observation = {'position': np.zeros((2,))-1e6, 'heading': float(-1e6), 'type': sim_type, 'observed_agent': observed_agent}

    if sim_type == 'self' or 'sim': # Assume no noise for self-observation or simulated observation
        observation['position'] = true_position[:2]
        observation['heading'] = true_position[2]       
    else:
        observation['position'] = true_position[:2] + np.random.normal(0, observation_error_std, true_position[:2].shape)
        observation['heading'] = true_position[2] + np.random.normal(0, observation_error_std*0.1) 

    return observation

def calculate_kl_divergence(p, q):
    """Calculate KL divergence between two probability distributions."""
    return np.sum(p * log_stable(p / (q + np.exp(-16))))/16 #Divide by 16 to normalize the KL divergence

def calculate_shannon_entropy(p):
    """Calculate Shannon entropy of a probability distribution."""
    return -np.sum(p * log_stable(p))

def get_likelihood(robot_pose,observations,goals,max_distance_measure=1.0,system_dict=None, use_ep=False, predict_self=False):
    """Calculate the likelihood of a goal being the target based on agent positions."""
    likelihood = np.zeros(goals.shape[0]).flatten()
    for observation in observations:
        likelihood += salience(robot_pose,observation, goals,max_distance_measure,system_dict, use_ep, predict_self).flatten()
    return softmax(likelihood)

import numpy as np

def salience(robot_pose, observation, goals, max_distance_measure=1.0, system_dict=None, use_ep=False, predict_self=False):
    """
    Calculate the salience (importance) of each goal based on depth and azimuth measurements from the robot.

    Parameters:
    robot_pose (ndarray): The current position and heading of the robot.
    observation (dict): The observation data containing position, heading, and type.
    goals (ndarray): The positions of potential goals.
    max_distance_measure (float): The maximum distance measure for normalizing depth differences.
    system_dict (dict): System-specific parameters and configurations.
    use_ep (bool): Flag to indicate whether to use an epistemic policy.
    predict_self (bool): Flag to indicate if the observation is a self-prediction.

    Returns:
    ndarray: An array of salience values for each goal.
    """
    # Compute vectors from the robot to each goal
    goal_vectors = goals - robot_pose[:2]
    
    # Compute azimuth angles from the robot to each goal
    goal_azimuths = np.arctan2(goal_vectors[:, 1], goal_vectors[:, 0])
    
    # Initialize salience scores for each goal
    saliences = np.zeros(len(goals))

    if observation['type'] in ['A'] or predict_self:
        # Calculate depth differences between observed position and each goal
        depth_differences = np.linalg.norm(observation['position'] - goals, axis=1)
        
        # Compute depth salience with an exponential decay function
        depth_salience = np.exp(-depth_differences / max_distance_measure)
        
        # Add depth salience to the total salience score
        saliences += depth_salience
    
    if observation['type'] in ['A', 'B'] or predict_self:
        # Calculate observed azimuth angle from robot to the observed position
        observed_azimuth = np.arctan2(observation['position'][1] - robot_pose[1], observation['position'][0] - robot_pose[0])
        
        # Calculate relative azimuth angles between each goal azimuth and the observed azimuth
        relative_azimuths = np.abs((goal_azimuths + observed_azimuth + np.pi) % (2 * np.pi) - np.pi)
        
        # Compute azimuth salience, normalizing and inverting to prioritize smaller angles
        azimuth_salience = 1.0 / 8 * np.exp(-relative_azimuths / np.pi)
        
        # Add azimuth salience to the total salience score
        saliences += azimuth_salience
        
        # Calculate heading angles from observed position to each goal
        heading_to_goal = (np.arctan2(goals[:, 1] - observation['position'][1], goals[:, 0] - observation['position'][0]) - observation['heading'] + np.pi) % (2 * np.pi) - np.pi
        
        # Compute heading salience with exponential decay based on heading difference
        heading_salience = 1.0 / 8 * np.exp(-np.abs(heading_to_goal) / np.pi)
        
        # Add heading salience to the total salience score
        saliences += heading_salience

    if observation['type'] != 'self' and use_ep:
        # If the observation is not self and epistemic policy is used
        which_agent = observation['observed_agent']
        
        # Check if the observed agent type matches the conditions
        if (system_dict['agent_types'][which_agent] == 'B' and observation['type'] == 'A') or (system_dict['agent_types'][which_agent] == observation['type']):
            # Compute goal vectors and azimuths based on observed agent's position
            goal_vectors = goals - observation['position']
            goal_azimuths = np.arctan2(goal_vectors[:, 1], goal_vectors[:, 0])
            
            # Calculate observed azimuth angle from observed position to the robot
            observed_azimuth = np.arctan2(robot_pose[1] - observation['position'][1], robot_pose[0] - observation['position'][0])
            
            # Calculate relative azimuth angles between each goal azimuth and the observed azimuth
            relative_azimuths = np.abs((goal_azimuths + observed_azimuth + np.pi) % (2 * np.pi) - np.pi)
            
            # Compute azimuth salience for the observed agent
            azimuth_salience = 1.0 / 8 * np.exp(-relative_azimuths / np.pi)
            
            # Add azimuth salience to the total salience score
            saliences += azimuth_salience
            
            # Calculate heading angles from observed position to each goal
            heading_to_goal = (np.arctan2(goals[:, 1] - observation['position'][1], goals[:, 0] - observation['position'][0]) - observation['heading'] + np.pi) % (2 * np.pi) - np.pi
            
            # Compute heading salience for the observed agent
            heading_salience = 1.0 / 8 * np.exp(-np.abs(heading_to_goal) / np.pi)
            
            # Add heading salience to the total salience score
            saliences += heading_salience

    return saliences



def predict_agent_position(agent_position, velocity, heading):
    """
    Predict the agent's next position based on the given velocity and heading.
    
    Parameters:
    agent_position (ndarray): Current position and heading of the agent.
    velocity (float): Chosen velocity for the agent.
    heading (float): Chosen heading for the agent.

    Returns:
    ndarray: Predicted position and heading of the agent.
    """
    # Make a copy of the current position to avoid modifying the original array
    agent_prediction = np.copy(agent_position)
    
    # Update the x-coordinate based on velocity and heading
    agent_prediction[0] += velocity * np.cos(heading)
    
    # Update the y-coordinate based on velocity and heading
    agent_prediction[1] += velocity * np.sin(heading)
    
    # Update the heading
    agent_prediction[2] = heading
    
    return agent_prediction


def parse_observations(observations):
    """
    Parse observations to extract agent positions and headings.
    
    Parameters:
    observations (list): List of observation dictionaries, each containing 'position' and 'heading' keys.

    Returns:
    ndarray: Array of agent positions and headings.
    """
    # Initialize an array to store positions and headings
    agent_positions = np.zeros((len(observations), len(observations[0]['position']) + 1))
    
    # Iterate over observations and extract positions and headings
    for idx, observation in enumerate(copy.deepcopy(observations)):
        agent_positions[idx] = np.concatenate((observation['position'], [observation['heading']]), axis=0)
    
    return agent_positions.copy()

def make_decision(args, system_dict, use_ep=True):
    """
    Agent decision-making based on active inference to encourage convergence on a shared goal.
    
    Parameters:
    args (dict): Dictionary containing various arguments for decision-making.
    system_dict (dict): System-specific parameters and configurations.
    use_ep (bool): Flag to indicate whether to use an epistemic policy.

    Returns:
    tuple: Best action, updated observations, and goal scores.
    """
    # Parse arguments
    agent_id = args['agent_id']
    agent_type = args['agent_type']
    agent_positions = args['agent_positions']
    prior = args['prior']
    goals = args['goals']
    velocity_options = args['velocity_options']
    heading_options = args['heading_options']
    max_distance_measure = args['max_distance_measure']
    observation_error_std = args['observation_error_std']

    # Initialize variables for decision-making
    best_action = None
    best_score = np.inf
    goal_scores = []
    observations = []

    # Start decision loops for each possible action
    for idx in range(agent_positions.shape[0]):
        if idx == agent_id:
            # Simulate observation for the self agent
            observations.append(simulate_observation(agent_positions[idx], 0, 'self', idx))
        else:
            # Simulate observation for other agents
            observations.append(simulate_observation(agent_positions[idx], observation_error_std, agent_type, idx))

    # Parse observed positions from observations
    observed_positions = parse_observations(observations)
    
    # Iterate through each velocity and heading option to evaluate possible actions
    for velocity in velocity_options:
        for heading in heading_options:
            # Predict the new position of the agent based on current velocity and heading
            predicted_position = predict_agent_position(observed_positions[agent_id], velocity, heading)
            
            # Update the positions array with the predicted position for the agent
            new_agent_positions = np.copy(agent_positions)
            new_agent_positions[agent_id] = predicted_position
            
            # Update the observation for the agent with the predicted position and heading
            predicted_observation = copy.deepcopy(observations)
            predicted_observation[agent_id]['position'] = np.copy(predicted_position[:2])
            predicted_observation[agent_id]['heading'] = predicted_position[2]
            
            # Calculate the posterior likelihood of the agent reaching the goals
            posterior = get_likelihood(observed_positions[agent_id], predicted_observation, goals, max_distance_measure, system_dict, use_ep, predict_self=True)
            
            # Calculate the Kullback-Leibler divergence between prior and posterior
            kl_divergence = calculate_kl_divergence(prior, posterior)
            
            # Calculate the Shannon entropy of the posterior
            entropy = calculate_shannon_entropy(posterior)
            
            # Append the calculated scores and actions to the goal_scores list
            goal_scores.append((entropy + kl_divergence, velocity, heading, entropy, kl_divergence, posterior))
        
    # Choose the action that minimizes the combined distance for the current goal
    best_action_for_goal = min(goal_scores, key=lambda x: x[0])

    # Update the best action if the current goal is more attainable than the previous best
    if best_action_for_goal[0] < best_score:
        best_score = best_action_for_goal[0]
        best_action = best_action_for_goal[1], best_action_for_goal[2]
    
    return best_action, observations, goal_scores

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
        agent_vars.append(agent_dict)
    return agent_vars


def run_simulation(args, max_iterations=100):
    """
    Run the simulation until each agent converges to a different goal or maximum iterations are reached.

    Parameters:
    args (dict): Dictionary containing simulation parameters and initial states.
    max_iterations (int): Maximum number of iterations for the simulation.

    Returns:
    tuple: Current positions of agents, list of selected goals (or None), iteration count, return arguments, 
           final priors for each agent, and free energy scores.
    """
    # Initialize return arguments and variables
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

    # Run the simulation for a maximum of max_iterations
    for iteration in range(max_iterations):
        # Print a message if maximum iterations are reached
        if iteration == max_iterations - 1:
            print("Max iterations reached.")
        
        # Initialize lists to store decisions, observations, and free energy scores
        decisions = []
        observations = []

        # Make decisions for all agents
        for idx in range(num_agents):
            decision, observation, free_energy_score = make_decision(agent_vars[idx], args, use_ep)  # Make decision for agent
            decisions.append(decision)  # Save decisions for all agents
            observations.append(observation)  # Save observations for all agents
            free_energy_scores.append((free_energy_score, idx, iteration))  # Save free energy scores for all agents

        # Update agent positions based on their decisions
        for agent_id, (velocity, heading) in enumerate(decisions):
            dx = velocity * np.cos(heading)
            dy = velocity * np.sin(heading)
            true_positions[agent_id] += np.array([dx, dy, 0])  # Update position
            true_positions[agent_id][2] = heading  # Update heading

        # Update agent variables with new positions and observations
        for agent_id in range(num_agents):
            agent_vars[agent_id]['agent_positions'] = np.copy(true_positions)
            observations[agent_id][agent_id]['position'] = true_positions[agent_id][:2]
            observations[agent_id][agent_id]['heading'] = true_positions[agent_id][2]
            agent_vars[agent_id]['prior'] = get_likelihood(
                observations[agent_id][agent_id]['position'],
                observations[agent_id], 
                goals,
                args['max_distance_measure'],
                args,
                use_ep
            )

        # Check if agents have selected different goals
        current_positions = np.copy(true_positions)
        distances_to_goals = [np.linalg.norm(goals - pos[:2], axis=1) for pos in current_positions]
        
        # Initialize list to track the best goal for each agent
        best_goals = [-1] * num_agents
        
        for agent_id in range(num_agents):
            likelihoods = []
            for goal_idx in range(len(goals)):
                # Compute likelihood of other agents choosing this goal
                other_agents_likelihood = 0
                for other_agent_id in range(num_agents):
                    if other_agent_id != agent_id:
                        other_agents_likelihood += get_goal_likelihood(agent_vars[other_agent_id], goals[goal_idx], other_agent_id)
                likelihoods.append(other_agents_likelihood)
            
            # Select the goal with the lowest likelihood of being chosen by other agents
            
            best_goal = np.argmin(likelihoods)
            best_goals[agent_id] = best_goal
            
        # Ensure each agent has a unique goal
        unique_selected_goal = []
        for goal in best_goals:
            if goal not in unique_selected_goal:
                unique_selected_goal.append(goal)
            else:
                # If goal is already selected by another agent, find the next least likely goal
                sorted_likelihoods = np.argsort(likelihoods)
                for alternative_goal in sorted_likelihoods:
                    if alternative_goal not in unique_selected_goal:
                        unique_selected_goal.append(alternative_goal)
                        break

        selected_goal = unique_selected_goal
        for i in range(num_agents):
            print('Selected Goal for ', i, ' is ', selected_goal[i])
        
        # Check convergence conditions
        all_selected = len(set(selected_goal)) == num_agents
        distances_to_selected_goal = [distances[goal] for distances, goal in zip(distances_to_goals, selected_goal)]

        # Save current positions for plotting
        return_args['positions'].append(np.copy(current_positions))
        
        if (np.array(distances_to_selected_goal) < 1.2).all() and all_selected:
            converged_count += 1
            if converged_count >= 5:
                print(f"Agents have converged to different goals {selected_goal} after {iteration + 1} iterations.")
                return (current_positions, selected_goal, iteration, return_args, 
                        [agent_vars[i]['prior'] for i in range(num_agents)], free_energy_scores)

    print("Agents did not converge to different goals within the maximum iterations.")
    for i in range(num_agents):
        print('Final Selected Goal for robot', i, ' is ', selected_goal[i])
    return (current_positions, None, iteration, return_args, 
            [agent_vars[i]['prior'] for i in range(num_agents)], free_energy_scores)

def get_goal_likelihood(agent_var, goal, other_agent_id, max_distance_measure=5.0):
    """
    Compute the likelihood of an agent choosing a particular goal.

    Parameters:
    agent_var (dict): Agent-specific variables, including current position and heading.
    goal (array): The goal position.
    max_distance_measure (float): Maximum distance measure for normalization.

    Returns:
    float: Likelihood of choosing the goal.
    """
    # Extract agent's current position and heading
    agent_position_tuple = agent_var['agent_positions'][other_agent_id]
    agent_position = agent_position_tuple[:2]  # Extract only x, y coordinates
    agent_heading = agent_position_tuple[2] #Extract just the heading
    
    # Calculate the Euclidean distance to the goal
    distance_to_goal = np.linalg.norm(agent_position - goal)

    # Calculate the angle to the goal from the agent's current position
    goal_direction = np.arctan2(goal[1] - agent_position[1], goal[0] - agent_position[0])
    heading_difference = np.abs((goal_direction - agent_heading + np.pi) % (2 * np.pi) - np.pi)

    # Normalize distance and heading difference to calculate likelihood
    distance_likelihood = np.exp(-distance_to_goal / max_distance_measure)  # Exponential decay based on distance
    heading_likelihood = np.exp(-heading_difference / np.pi)  # Exponential decay based on heading difference

    # Combine distance and heading likelihoods (weights can be adjusted)
    combined_likelihood = distance_likelihood * heading_likelihood
    #print('combined liklihood of ', other_agent_id, ' going to goal ', goal, ' is ', combined_likelihood)
    return combined_likelihood





class PlotSim:
    """Class to plot the simulation of agents moving towards a goal."""
    def __init__(self, num_agents, goals):
        # Initialize plotting objects
        self.fig, self.ax = plt.subplots()
        plt.xlim(-5, 15)
        plt.ylim(-5, 15)
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

def parse_free_energy_scores(free_energy_scores):
    """Parse the free energy scores for each agent over time."""
    res_list = []
    for entry in free_energy_scores:
        avg_scores = np.min([score[0] for score in entry[0]])
        agent = entry[1]
        iteration = entry[2]
        res_list.append([avg_scores, agent, iteration])
    return res_list

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


# Shell of code to be worked on here
def generate_likelihood_representations(robot_positions, goals, observation_error_std=2.0):
    """
    Generate likelihood representations for different combinations of robot-goal assignments.

    Parameters:
        robot_positions (list): List of robot positions, where each position is a tuple (x, y, theta).
        goals (list): List of goal positions, where each goal is a tuple (x, y).
        observation_error_std (float): Standard deviation of observation noise.

    Returns:
        dict: A dictionary where keys are combinations of robot-goal assignments
              and values are likelihood representations for each combination.
    """
    import itertools

    # Generate all possible combinations of robot-goal assignments
    robot_indices = range(len(robot_positions))
    goal_indices = range(len(goals))
    combinations = list(itertools.permutations(goal_indices, len(robot_positions)))

    # Initialize dictionary to store likelihood representations
    likelihood_representations = {}

    # Iterate over each combination of robot-goal assignments
    for combo in combinations:
        # Simulate observations for each robot based on its assigned goal
        observations = []
        for i, robot_index in enumerate(robot_indices):
            assigned_goal_index = combo[i]
            observation = simulate_observation(robot_positions[i], observation_error_std, 'A', assigned_goal_index)
            observations.append(observation)

        # Calculate likelihood representation for this combination of observations
        likelihood_representation = get_likelihood(robot_positions, observations, goals)

        # Store likelihood representation in dictionary
        likelihood_representations[combo] = likelihood_representation

    return likelihood_representations