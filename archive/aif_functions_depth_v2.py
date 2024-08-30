import numpy as np
from matplotlib.patches import Polygon, Circle
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set1_9
import copy
import pandas as pd

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum()

def log_stable(x):
    """Compute log values for each sets of scores in x."""
    return np.log(x + np.exp(-16))

def simulate_observation(true_position, observation_error_std=2.0, sim_type='A'):
    """Simulate noisy observation of another agent's position."""
    observation = {'position': np.zeros((2,))-1e6, 'vector': np.zeros((2,))-1e6, 'heading': float(-1e6), 'type': sim_type}

    if sim_type == 'self':
        observation['position'] = true_position[:2]
        observation['heading'] = true_position[2]
    elif sim_type == 'A':
        observation['position'] = true_position[:2] + np.random.normal(0, observation_error_std, true_position[:2].shape)
        observation['heading'] = true_position[2] + np.random.normal(0, observation_error_std*0.1) 
    elif sim_type == 'B':
        observation['vector'] = np.arctan2(true_position[1] - observation['position'][1], true_position[0] - observation['position'][0])
        observation['heading'] = true_position[2] + np.random.normal(0, observation_error_std*0.1)
    
    observation['type'] = sim_type
    return observation

def calculate_kl_divergence(p, q):
    """Calculate KL divergence between two probability distributions."""
    return np.sum(p * log_stable(p / (q + np.exp(-16))))/16 #Divide by 16 to normalize the KL divergence

def calculate_shannon_entropy(p):
    """Calculate Shannon entropy of a probability distribution."""
    return -np.sum(p * log_stable(p))

def get_likelihood(robot_pose,observations,goals,max_distance_measure=1.0):
    """Calculate the likelihood of a goal being the target based on agent positions."""
    likelihood = np.zeros(goals.shape[0]).flatten()
    for observation in observations:
        likelihood += salience(robot_pose,observation, goals,max_distance_measure).flatten()
    return softmax(likelihood)

def salience(robot_pose, observation, goals, max_distance_measure=1.0):
    """Calculate the salience of each goal based on depth and azimuth measurements from the robot."""
    goal_vectors = goals - robot_pose[:2]
    goal_azimuths = np.arctan2(goal_vectors[:, 1], goal_vectors[:, 0])
    saliences = np.zeros(len(goals))

    if observation['type'] in ['A','self']:
        # For depth, compute how close each goal is to the observed depth
        depth_differences = np.linalg.norm(observation['position'] - goals, axis=1)
        depth_salience = np.exp(-depth_differences / max_distance_measure)  # exponential decay based on depth difference
        saliences += depth_salience
    
    if observation['type'] in ['A', 'B', 'self']:
        # For azimuth, compute alignment of each goal with the observed azimuth
        observed_azimuth = np.arctan2(observation['position'][1]-robot_pose[1], observation['position'][0]-robot_pose[0])
        relative_azimuths = np.abs((goal_azimuths + observed_azimuth + np.pi) % (2 * np.pi) - np.pi)
        azimuth_salience = 1./8 * np.exp(- relative_azimuths / np.pi)  # normalize and invert to make smaller angles more salient
        saliences += azimuth_salience
        # Compute if observed robot is heading towards the goal
        heading_to_goal = (np.arctan2(goals[:, 1] - observation['position'][1], goals[:, 0] - observation['position'][0]) - observation['heading'] + np.pi) % (2 * np.pi) - np.pi
        heading_salience = 1./8 * np.exp(- np.abs(heading_to_goal) / np.pi)
        saliences += heading_salience

    return saliences


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

def make_decision(args):
    """Agent decision-making based on active inference to encourage convergence on a shared goal."""
    #Parse arguments
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
            observations.append(simulate_observation(agent_positions[idx], 0, 'self'))
        else:
            observations.append(simulate_observation(agent_positions[idx], observation_error_std, agent_type))

    observed_positions = parse_observations(observations)
    # Calculate the likelihood of each goal being the target based on agent measurements -> May be able to turn into NN
    for velocity in velocity_options:
        for heading in heading_options:
            predicted_position = predict_agent_position(observed_positions[agent_id], velocity, heading)
            new_agent_positions = np.copy(agent_positions)
            new_agent_positions[agent_id] = predicted_position
            predicted_observation = copy.deepcopy(observations)
            predicted_observation[agent_id]['position'] = np.copy(predicted_position[:2])
            predicted_observation[agent_id]['heading'] = predicted_position[2]
            #Predict how what you would do would differ from the prior (what the system is doing - complexity)
            posterior = get_likelihood(observed_positions[agent_id], predicted_observation, goals, max_distance_measure)
            kl_divergence = calculate_kl_divergence(prior, posterior)
            # kl_divergence = 0.0
            entropy = calculate_shannon_entropy(posterior)
            # Estimate how both agents are aligned with reaching the current goal                
            goal_scores.append((entropy + kl_divergence, velocity, heading,  entropy, kl_divergence, posterior))
        
    # Choose the action (for the current goal) that minimizes the combined distance
    best_action_for_goal = min(goal_scores, key=lambda x: x[0])

    # Update best action if this goal is more attainable than previous best
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
    """Run the simulation until both agents converge to the same goal or max iterations reached."""
    return_args = {}
    agent_vars = parse_args_by_agent(args)
    num_agents = len(args['agent_positions'])
    current_positions = np.copy(args['agent_positions'])
    true_positions = np.copy(current_positions)
    goals = args['goals']
    return_args['positions'] = [np.copy(current_positions)]
    free_energy_scores = []

    for iteration in range(max_iterations):
        # Make decisions for all agents
        if iteration == max_iterations - 1:
            print("Max iterations reached.")
        decisions = []
        observations = []

        for idx in range(num_agents):
            decision, observation, free_energy_score = make_decision(agent_vars[idx]) # Make decision for agent
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
            agent_vars[agent_id]['prior'] = get_likelihood(observations[agent_id][agent_id]['position'],observations[agent_id], 
                                                           goals,args['max_distance_measure'])
                    
        # Check if agents have converged to the same goal
        current_positions = np.copy(true_positions)
        distances_to_goals = [np.linalg.norm(goals - pos[:2], axis=1) for pos in current_positions]
        distances_to_selected_goal = [np.min(distances) for distances in distances_to_goals]
        selected_goal = [np.argmin(distances) for distances in distances_to_goals]
        all_same_goal = (selected_goal[0] == which_goal for which_goal in selected_goal)
        
        # Save current positions for plotting
        return_args['positions'].append(np.copy(current_positions))
        if (np.array(distances_to_selected_goal)<1.0).all() and all_same_goal:
            print(distances_to_selected_goal)
            print(f"Agents have converged to Goal {selected_goal[0]} after {iteration + 1} iterations.")
            return current_positions, selected_goal[0], iteration, return_args, [agent_vars[i]['prior'] for i in range(num_agents)], free_energy_scores

    print("Agents did not converge to the same goal within the maximum iterations.")
    return current_positions, None, iteration, return_args, [agent_vars[i]['prior'] for i in range(num_agents)], free_energy_scores

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
    plt.show()