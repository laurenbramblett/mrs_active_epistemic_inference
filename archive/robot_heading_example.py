import numpy as np, matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from palettable.colorbrewer.qualitative import Set1_9
from pathlib import Path
from datetime import datetime
matplotlib.use('TkAgg')  # Use an interactive backend for animation

pwd = str(Path(__file__).parent.absolute()) + "/"
cmap = Set1_9.mpl_colors
# Re-define the environment and simulation parameters here
goals = np.array([[0, 0], [10, 10]], dtype=float)  # Goal positions
agent_positions = np.array([[1, 1], [8, 9]], dtype=float)  # Initial agent positions
velocity_options = [0, 0.5, 1.0]  # Velocity options for the agents
heading_options = [0, np.pi/4, np.pi, 3*np.pi/4]  # Heading options (radians)
observation_error_std = 5.0  # Observation noise standard deviation
max_iterations = 100  # Maximum number of iterations

# Initialize figure for plotting
fig, ax = plt.subplots()
plt.xlim(-5, 15)
plt.ylim(-5, 15)
# Paths (with smaller line width)
agent_paths = [ax.plot([], [], 'o-', markersize=3, linewidth=1, alpha=0.5, color=cmap[i])[0] for i in range(2)]
# Current positions (with larger markers)
agent_markers = [ax.plot([], [], 'o', markersize=10, color=cmap[i])[0] for i in range(2)]
goal_plots = [ax.plot(goal[0], goal[1], 'x', markersize=10, color='purple')[0] for goal in goals]  # Plot goals

def init():
    """Initialize the background of the plot."""
    for agent_path, agent_marker in zip(agent_paths, agent_markers):
        agent_path.set_data([], [])
        agent_marker.set_data([], [])
    return agent_paths + agent_markers

def update(frame):
    """Update the plot for each frame."""    
    decisions = [make_decision(agent_id, agent_positions) for agent_id in range(2)]
    
    # Update positions based on decisions
    for agent_id, (velocity, heading) in enumerate(decisions):
        dx = velocity * np.cos(heading)
        dy = velocity * np.sin(heading)
        agent_positions[agent_id] += np.array([dx, dy])
    
    # Update plot data
    for agent_id, (agent_path, agent_marker) in enumerate(zip(agent_paths, agent_markers)):
        xdata, ydata = agent_path.get_data()
        xnew, ynew = agent_positions[agent_id]
        xdata = np.append(xdata, xnew)
        ydata = np.append(ydata, ynew)
        agent_path.set_data(xdata, ydata)
        agent_marker.set_data(xnew, ynew)
    
    return agent_paths + agent_markers


def simulate_observation(true_position):
    """Simulate noisy observation of another agent's position."""
    observed_position = true_position + np.random.normal(0, observation_error_std, true_position.shape)
    return observed_position

def calculate_kl_divergence(p, q):
    """Calculate KL divergence between two probability distributions."""
    return np.sum(p * np.log(p / q + np.exp(-16)))

def calculate_shannon_entropy(p):
    """Calculate Shannon entropy of a probability distribution."""
    return -np.sum(p * np.log(p))

def predict_agent_position(agent_position, velocity, heading):
    """Predict agent's next position based on chosen velocity and heading."""
    dx = velocity * np.cos(heading)
    dy = velocity * np.sin(heading)
    return agent_position + np.array([dx, dy])

def make_decision(agent_id, agent_positions):
    """Agent decision-making based on active inference to encourage convergence on a shared goal."""
    best_action = None
    best_score = np.inf

    other_agent_observed_position = simulate_observation(agent_positions[1 - agent_id])

    for goal in goals:
        # Initialize a score for how attainable each goal seems for both agents
        goal_scores = []

        for velocity in velocity_options:
            for heading in heading_options:
                predicted_position = predict_agent_position(agent_positions[agent_id], velocity, heading)
                
                # Estimate how both agents are aligned with reaching the current goal
                distance_to_goal = np.linalg.norm(predicted_position - goal)
                distance_other_to_goal = np.linalg.norm(other_agent_observed_position - goal)

                # Use the sum of both distances as a simple score for this action's alignment with the goal
                goal_alignment_score = distance_to_goal + distance_other_to_goal
                
                goal_scores.append((goal_alignment_score, velocity, heading))

        # Choose the action (for the current goal) that minimizes the combined distance
        best_action_for_goal = min(goal_scores, key=lambda x: x[0])

        # Update best action if this goal is more attainable than previous best
        if best_action_for_goal[0] < best_score:
            best_score = best_action_for_goal[0]
            best_action = best_action_for_goal[1], best_action_for_goal[2]
    
    return best_action


def run_simulation(max_iterations=100):
    """Run the simulation until both agents converge to the same goal or max iterations reached."""
    current_positions = np.copy(agent_positions)
    
    for iteration in range(max_iterations):
        decisions = [make_decision(agent_id, current_positions) for agent_id in range(2)]
        
        # Update agent positions based on their decisions
        for agent_id, (velocity, heading) in enumerate(decisions):
            dx = velocity * np.cos(heading)
            dy = velocity * np.sin(heading)
            current_positions[agent_id] += np.array([dx, dy])
        
        # Check if agents have converged to the same goal
        distances_to_goals = [np.linalg.norm(goals - pos, axis=1) for pos in current_positions]
        for agent_id, distances in enumerate(distances_to_goals):
            print(f"Agent {agent_id} distances to goals: {distances}")
            if np.min(distances) < 1.0:
                print(f"Agent {agent_id} has reached Goal {np.argmin(distances)} after {iteration + 1} iterations.")
        goal_reached_by_agents = [np.argmin(distances) for distances in distances_to_goals]
        
        if (np.array(distances_to_goals)<1).all():
            print(f"Agents have converged to Goal {goal_reached_by_agents[0]} after {iteration + 1} iterations.")
            return current_positions, goal_reached_by_agents[0]

    print("Agents did not converge to the same goal within the maximum iterations.")
    return current_positions, None

# Run the simulation
final_positions, goal_converged = run_simulation()
# Create animation
ani = FuncAnimation(fig, update, frames=range(max_iterations), init_func=init, blit=True, repeat=False)

# Save the animation as a video
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
ani.save(pwd + "videos/two_goals_choice" + current_time + ".mp4", writer='ffmpeg', fps=20, dpi=300)
print("Image saved as: ", pwd + "videos/two_goals_choice" + current_time + ".mp4")
