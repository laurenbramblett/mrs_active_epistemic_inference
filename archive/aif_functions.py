import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum()

def log_stable(x):
    """Compute log values for each sets of scores in x."""
    return np.log(x + np.exp(-16))

def simulate_observation(true_position, observation_error_std=2.0):
    """Simulate noisy observation of another agent's position."""
    observed_position = true_position + np.random.normal(0, observation_error_std, true_position.shape)
    observed_position[2] = np.max((np.min((observed_position[2], np.pi)), -np.pi))
    return observed_position

def calculate_kl_divergence(p, q):
    """Calculate KL divergence between two probability distributions."""
    return np.sum(p * log_stable(p / (q + np.exp(-16))))

def calculate_shannon_entropy(p):
    """Calculate Shannon entropy of a probability distribution."""
    return -np.sum(p * log_stable(p))

def get_likelihood(agent_positions,goals,beliefs,max_distance_measure=1.0):
    """Calculate the likelihood of a goal being the target based on agent positions."""
    likelihood = np.zeros(goals.shape[0]).flatten()
    for agent_position in agent_positions:
        likelihood += salience(agent_position, goals, beliefs,max_distance_measure).flatten()
    return softmax(likelihood)

def salience(agent_position, goals, beliefs, max_distance_measure=1.0):
    """Calculate the salience of each goal based on agent's current position."""
    salience = np.zeros(goals.shape[0])
    peakiness = 1.0  # Peakiness parameter
    for i, goal in enumerate(goals):
        distance_measure = np.clip(np.linalg.norm(goal - agent_position[:2]), 0, max_distance_measure)
        heading_measure = ((np.arctan2(goal[1] - agent_position[1], goal[0] - agent_position[0]) - agent_position[2]) + np.pi) % (2 * np.pi) - np.pi
        salience[i] = (max_distance_measure - distance_measure)/max_distance_measure + 1./4 * (np.pi - np.abs(heading_measure))/np.pi
    return salience


def predict_agent_position(agent_position, velocity, heading):
    """Predict agent's next position based on chosen velocity and heading."""
    predicted_position = np.copy(agent_position)
    predicted_position[0] += velocity * np.cos(heading)
    predicted_position[1] += velocity * np.sin(heading)
    predicted_position[2] = heading
    return predicted_position

def make_decision(agent_id, agent_positions, prior, goals, velocity_options, heading_options, max_distance_measure=1.0, observation_error_std=2.0):
    """Agent decision-making based on active inference to encourage convergence on a shared goal."""
    best_action = None
    best_score = np.inf

    goal_scores = []    
    observed_positions = np.copy(agent_positions)
    for id in range(agent_positions.shape[0]):
        if id == agent_id:
            observed_positions[id] = agent_positions[id]
        else:
            observed_positions[id] = simulate_observation(agent_positions[id], observation_error_std)

    for velocity in velocity_options:
        for heading in heading_options:
            predicted_position = predict_agent_position(observed_positions[agent_id], velocity, heading)
            new_agent_positions = np.copy(observed_positions)
            new_agent_positions[agent_id] = predicted_position
            #Predict how what you would do would differ from the prior (what the system is doing - complexity)
            posterior = get_likelihood(new_agent_positions, goals, prior, max_distance_measure)
            kl_divergence = calculate_kl_divergence(prior, posterior)
            # kl_divergence = 0.0
            entropy = calculate_shannon_entropy(posterior)
            # Estimate how both agents are aligned with reaching the current goal                
            goal_scores.append((entropy + kl_divergence, velocity, heading))
        
    # Choose the action (for the current goal) that minimizes the combined distance
    best_action_for_goal = min(goal_scores, key=lambda x: x[0])

    # Update best action if this goal is more attainable than previous best
    if best_action_for_goal[0] < best_score:
        best_score = best_action_for_goal[0]
        best_action = best_action_for_goal[1], best_action_for_goal[2]
    
    return best_action

def run_simulation(args, max_iterations=100):
    """Run the simulation until both agents converge to the same goal or max iterations reached."""
    goals = args['goals']
    agent_positions = args['agent_positions']
    num_agents = len(agent_positions)
    velocity_options = args['velocity_options']
    heading_options = args['heading_options']
    max_distance_measure = args['max_distance_measure']
    observation_error_std = args['observation_error_std']
    return_args = {}
    prior = np.repeat(np.repeat(1/len(goals), len(goals)), num_agents).reshape(num_agents, len(goals))
    print(prior[0])
    current_positions = np.copy(agent_positions)
    return_args['positions'] = [np.copy(agent_positions)]
    
    for iteration in range(max_iterations):
        # Make decisions for all agents
        if iteration == max_iterations - 1:
            print("Max iterations reached.")
        decisions = [make_decision(agent_id, current_positions, prior[agent_id], goals, velocity_options, 
                                   heading_options, max_distance_measure, observation_error_std) for agent_id in range(num_agents)]
        
        # Update agent positions based on their decisions
        for agent_id, (velocity, heading) in enumerate(decisions):
            dx = velocity * np.cos(heading)
            dy = velocity * np.sin(heading)
            current_positions[agent_id] += np.array([dx, dy, 0])
            current_positions[agent_id][2] = heading
        decided_velocities = [decision[0] for decision in decisions]
        prior = [get_likelihood(current_positions, goals, prior[agent_id],max_distance_measure) for agent_id in range(num_agents)]
        
        # Check if agents have converged to the same goal
        distances_to_goals = [np.linalg.norm(goals - pos[:2], axis=1) for pos in current_positions]
        goal_reached_by_agents = [np.argmin(distances) for distances in distances_to_goals]
        distances_to_selected_goal = [np.min(distances) for distances in distances_to_goals]
        
        # Save current positions for plotting
        return_args['positions'].append(np.copy(current_positions))
        if (np.array(distances_to_selected_goal)<1.0).all() and decided_velocities == [0]*num_agents:
            print(distances_to_selected_goal)
            print(f"Agents have converged to Goal {goal_reached_by_agents[0]} after {iteration + 1} iterations.")
            return current_positions, goal_reached_by_agents[0], iteration, return_args, prior

    print("Agents did not converge to the same goal within the maximum iterations.")
    return current_positions, None, iteration, return_args, prior


def init_rosbot(ax, scale=0.1, color='r'):
    # Initialize parts of the Rosbot with placeholders for their geometric data
    elements = {
        'red_box': ax.fill([], [], color=color, alpha=0.3)[0],
        'left_tires': ax.fill([], [], color='k')[0],
        'right_tires': ax.fill([], [], color='k')[0],
        'light_bar': ax.fill([], [], color='k', alpha=0.5)[0],
    }
    return elements

def update_rosbot(elements, position, heading_ang, color, scale=0.3):
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
    R = np.array([
        [np.cos(heading_ang), -np.sin(heading_ang)],
        [np.sin(heading_ang), np.cos(heading_ang)]
    ])
    for name, part in parts.items():
        scaled_part = rosbot_scale * part
        transformed = R @ scaled_part + np.array(position)[:, None]
        elements[name].set_xy(transformed.T)
        if name in ['red_box']:
            elements[name].set_facecolor(color)  # Apply color only to these parts

