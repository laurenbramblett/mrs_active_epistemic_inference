import numpy as np
from matplotlib.patches import Polygon, Circle
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set1_9
import copy
import pandas as pd
import itertools
import random

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

def get_likelihood(robot_pose,observations,goals,max_distance_measure=1.0,system_dict=None, use_ep=False, predict_self = False):
    """Calculate the likelihood of a goal being the target based on agent positions."""
    likelihood = np.zeros(goals.shape[0]).flatten()
    for observation in observations:
        likelihood += salience(robot_pose,observation, goals,max_distance_measure,system_dict, use_ep, predict_self).flatten()
    return softmax(likelihood)

def salience(robot_pose, observation, goals, max_distance_measure=1.0, system_dict=None, use_ep=False, predict_self = False):
    """Calculate the salience of each goal based on depth and azimuth measurements from the robot."""
    goal_vectors = goals - robot_pose[:2]
    goal_azimuths = np.arctan2(goal_vectors[:, 1], goal_vectors[:, 0])
    saliences = np.zeros(len(goals))

    if observation['type'] in ['A'] or predict_self:
        # For depth, compute how close each goal is to the observed depth
        depth_differences = np.linalg.norm(observation['position'] - goals, axis=1)
        depth_salience = np.exp(-depth_differences / max_distance_measure)  # exponential decay based on depth difference
        saliences += depth_salience
    
    if observation['type'] in ['A', 'B'] or predict_self:
        # For azimuth, compute alignment of each goal with the observed azimuth
        observed_azimuth = np.arctan2(observation['position'][1]-robot_pose[1], observation['position'][0]-robot_pose[0])
        relative_azimuths = np.abs((goal_azimuths - observed_azimuth + np.pi) % (2 * np.pi) - np.pi)
        azimuth_salience = 1./8 * np.exp(- relative_azimuths / np.pi)  # normalize and invert to make smaller angles more salient
        saliences += azimuth_salience
        # Compute if observed robot is heading towards the goal
        heading_to_goal = (np.arctan2(goals[:, 1] - observation['position'][1], goals[:, 0] - observation['position'][0]) - observation['heading'] + np.pi) % (2 * np.pi) - np.pi
        heading_salience = 1./8 * np.exp(- np.abs(heading_to_goal) / np.pi)
        saliences += heading_salience

    if observation['type'] != 'self' and use_ep: # If the observation is not self, compute salience of self based on the observed agent's pose
        which_agent = observation['observed_agent']
        if (system_dict['agent_types'][which_agent] == 'B' and observation['type'] == 'A'): # or (system_dict['agent_types'][which_agent] == observation['type']):
            goal_vectors = goals - observation['position'] # Use the observed agent's position instead of the robot's
            goal_azimuths = np.arctan2(goal_vectors[:, 1], goal_vectors[:, 0]) # Recompute goal azimuths based on observed agent's position
            observed_azimuth = np.arctan2(robot_pose[1]-observation['position'][1], robot_pose[0]-observation['position'][0])
            relative_azimuths = np.abs((goal_azimuths - observed_azimuth + np.pi) % (2 * np.pi) - np.pi)
            azimuth_salience_ep = 1./8 * np.exp(- relative_azimuths / np.pi)  # normalize and invert to make smaller angles more salient
            saliences += azimuth_salience_ep
            # Compute if observed robot is heading towards the goal
            heading_to_goal = (np.arctan2(goals[:, 1] - observation['position'][1], goals[:, 0] - observation['position'][0]) - observation['heading'] + np.pi) % (2 * np.pi) - np.pi
            heading_salience_ep = 1./8 * np.exp(- np.abs(heading_to_goal) / np.pi)
            saliences += heading_salience_ep
            

    return saliences

def predict_agent_position(agent_position, velocity, heading):
    """Predict agent's next position based on chosen velocity and heading."""
    agent_prediction = np.copy(agent_position)
    agent_prediction[0] += velocity * np.cos(heading)
    agent_prediction[1] += velocity * np.sin(heading)
    agent_prediction[2] = heading
    return agent_prediction

def evaluate_state(prior, posterior):
    """Evaluate the desirability of a state based on the posterior probability of goals."""
    # Here we use negative entropy as a simple scoring mechanism: lower entropy means higher certainty/quality of the state.
    entropy = calculate_shannon_entropy(posterior)
    kl_divergence = calculate_kl_divergence(prior,posterior)
    return entropy + kl_divergence

# This function can be changed -- recommend using a neural network to predict the likelihood of a goal being the target
def heuristic_action(agent_position, goals, args):
    """Choose an action based on a simple heuristic: move towards the closest goal."""
    goal_directions = goals - agent_position[:2]
    goal_distances = np.linalg.norm(goal_directions, axis=1)
    closest_goal_idx = np.argmin(goal_distances)
    closest_goal_distance = goal_distances[closest_goal_idx]
    closest_velocity_option = np.argmin(np.abs(args['velocity_options']-closest_goal_distance))
    heading_to_goal = np.arctan2(goal_directions[closest_goal_idx][1], goal_directions[closest_goal_idx][0])
    # This is a simplistic approach; refine it based on your specific scenario
    return np.min(args['velocity_options'][closest_velocity_option]), heading_to_goal

# TODO: make more general for many different rollouts
def extended_greedy_rollout(args, system_dict, num_rollout_steps=1):
    """Perform a greedy rollout considering the actions of all agents."""
    agent_id = args['agent_id']
    current_positions = np.copy(args['agent_positions'])
    goals = args['goals']
    prior = args['prior']
    all_actions = list(itertools.product(args['velocity_options'], args['heading_options']))
    
    best_action = None
    best_score = float('-inf')
    goal_scores = []
    # Consider all possible actions for the focal agent
    for action in all_actions:
        total_score = 0
        
        for step in range(num_rollout_steps):
            simulated_positions = np.copy(current_positions)
            # Assume each agent, including the focal agent, might take any action
            for other_agent_id in range(len(simulated_positions)):
                velocity, heading = heuristic_action(simulated_positions[other_agent_id], goals, args) if other_agent_id != agent_id else action
                predicted_position = predict_agent_position(simulated_positions[other_agent_id], velocity, heading)
                simulated_positions[other_agent_id] = predicted_position

                predicted_position = predict_agent_position(simulated_positions[other_agent_id], velocity, heading)
                simulated_positions[other_agent_id] = predicted_position
            
            # Evaluate the system state after all agents have moved
            predicted_observations = [simulate_observation(pos, args['observation_error_std'], system_dict['agent_types'][idx], idx) for idx, pos in enumerate(simulated_positions)]
            posteriors = [get_likelihood(simulated_positions[i], predicted_observations, goals, args['max_distance_measure'], system_dict, args['use_ep'], predict_self=True) for i in range(len(simulated_positions))]
            system_score = sum(evaluate_state(prior,posterior) for posterior in posteriors)
            total_score += system_score
        goal_scores.append((total_score, action, posteriors.copy()))
        # Update best action based on the highest cumulative score over the rollout steps
        if total_score > best_score:
            best_score = total_score
            best_action = action

    return best_action, goal_scores


def parse_observations(observations):
    """Parse observations to get agent positions."""
    agent_positions = np.zeros((len(observations), len(observations[0]['position']) + 1))
    for idx, observation in enumerate(copy.deepcopy(observations)):
        agent_positions[idx] = np.concatenate((observation['position'], [observation['heading']]),axis=0)
    return agent_positions.copy()

def make_decision(args,system_dict, use_ep=True):
    """Agent decision-making based on active inference to encourage convergence on a shared goal."""
    #Parse arguments
    agent_id = args['agent_id']
    agent_type = args['agent_type']
    agent_positions = args['agent_positions']
    observation_error_std = args['observation_error_std']

    # Initialize all observations
    observations = []

    # Start decision loops for each possible action
    for idx in range(agent_positions.shape[0]):
        if idx == agent_id:
            observations.append(simulate_observation(agent_positions[idx], 0, 'self',idx))
        else:
            observations.append(simulate_observation(agent_positions[idx], observation_error_std, agent_type,idx))

    observed_positions = parse_observations(observations)
    args['agent_positions'] = observed_positions
    # Calculate the likelihood of each goal being the target based on agent measurements -> May be able to turn into NN
    best_action, goal_scores = extended_greedy_rollout(args, system_dict)
    
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
        agent_dict['use_ep'] = args['use_ep']
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
        if iteration == max_iterations - 1:
            print("Max iterations reached.")
        decisions = []
        observations = []
        if iteration > 176:
            s = 1

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
            agent_vars[agent_id]['prior'] = get_likelihood(observations[agent_id][agent_id]['position'],observations[agent_id], 
                                                           goals,args['max_distance_measure'],args, use_ep)
                    
        # Check if agents have converged to the same goal
        current_positions = np.copy(true_positions)
        distances_to_goals = [np.linalg.norm(goals - pos[:2], axis=1) for pos in current_positions]
        distances_to_selected_goal = [np.min(distances) for distances in distances_to_goals]
        selected_goal = [np.argmin(distances) for distances in distances_to_goals]
        all_same_goal = [selected_goal[0] == which_goal for which_goal in selected_goal]
        
        # Save current positions for plotting
        return_args['positions'].append(np.copy(current_positions))
        if (np.array(distances_to_selected_goal)<1.2).all() and all(all_same_goal):
            converged_count += 1
            if converged_count >= 5:
                print(f"Agents have converged to Goal {selected_goal[0]} after {iteration + 1} iterations.")
                return current_positions, selected_goal[0], iteration, return_args, [agent_vars[i]['prior'] for i in range(num_agents)], free_energy_scores

    print("Agents did not converge to the same goal within the maximum iterations.")
    return current_positions, None, iteration, return_args, [agent_vars[i]['prior'] for i in range(num_agents)], free_energy_scores

class PlotSim:
    """Class to plot the simulation of agents moving towards a goal."""
    def __init__(self, num_agents, goals, env_size=15):
        # Initialize plotting objects
        self.fig, self.ax = plt.subplots()
        plt.xlim(-5, env_size)
        plt.ylim(-5, env_size)
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
        avg_scores = np.min([score[0] for score in entry[0]])
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