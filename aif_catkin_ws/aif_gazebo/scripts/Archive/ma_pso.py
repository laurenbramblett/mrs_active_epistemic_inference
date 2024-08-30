import numpy as np
import matplotlib.pyplot as plt

def system_dynamics(x, u, time_step):
    return x + u * time_step

def fitness_function(position, initial_states, reference_states, num_agents, state_dim, prediction_horizon, time_step, max_velocity, min_distance):
    total_cost = 0
    position = position.reshape((num_agents, state_dim * (prediction_horizon + 1) + state_dim * prediction_horizon))
    
    # Initialize X and U
    X = np.zeros((num_agents, prediction_horizon + 1, state_dim))
    U = np.zeros((num_agents, prediction_horizon, state_dim))
    
    for i in range(num_agents):
        X[i] = position[i][:state_dim * (prediction_horizon + 1)].reshape((prediction_horizon + 1, state_dim))
        U[i] = position[i][state_dim * (prediction_horizon + 1):].reshape((prediction_horizon, state_dim))
        X[i, 0] = initial_states[i]
        
        for k in range(prediction_horizon):
            X[i, k + 1] = system_dynamics(X[i, k], U[i, k], time_step)
            prediction_error = np.sum((X[i, k] - reference_states[i]) ** 2)
            control_effort = np.sum((U[i, k]) ** 2)
            total_cost += prediction_error  # + 0.01 * control_effort
            if np.any(U[i, k] > max_velocity) or np.any(U[i, k] < -max_velocity):
                total_cost += 1e6  # High penalty for exceeding velocity

    # Collision avoidance
    for k in range(prediction_horizon):
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dist = np.linalg.norm(X[i, k] - X[j, k])
                if dist < min_distance:
                    total_cost += 1e6 * (min_distance - dist) ** 2  # High penalty for collisions

    return total_cost

def pso_optimize(initial_states, reference_states, num_agents, state_dim, prediction_horizon, time_step, max_velocity, min_distance, num_particles=30, max_iter=100):
    # PSO parameters
    w = 0.5
    c1 = 1.0
    c2 = 1.0

    dim = num_agents * (state_dim * (prediction_horizon + 1) + state_dim * prediction_horizon)
    
    # Initialize swarm
    swarm_positions = np.random.uniform(-max_velocity, max_velocity, (num_particles, dim))
    swarm_velocities = np.random.uniform(-max_velocity, max_velocity, (num_particles, dim))
    personal_best_positions = np.copy(swarm_positions)
    personal_best_scores = np.array([fitness_function(pos, initial_states, reference_states, num_agents, state_dim, prediction_horizon, time_step, max_velocity, min_distance) for pos in swarm_positions])
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    # PSO loop
    for iteration in range(max_iter):
        for i in range(num_particles):
            swarm_positions[i] += swarm_velocities[i]
            score = fitness_function(swarm_positions[i], initial_states, reference_states, num_agents, state_dim, prediction_horizon, time_step, max_velocity, min_distance)
            
            if score < personal_best_scores[i]:
                personal_best_positions[i] = swarm_positions[i]
                personal_best_scores[i] = score

            if score < global_best_score:
                global_best_position = swarm_positions[i]
                global_best_score = score

        # Update velocities
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            swarm_velocities[i] = (w * swarm_velocities[i] +
                                   c1 * r1 * (personal_best_positions[i] - swarm_positions[i]) +
                                   c2 * r2 * (global_best_position - swarm_positions[i]))

    best_position = global_best_position.reshape(num_agents, -1)
    return best_position, global_best_score

def plot_trajectories(initial_states, reference_states, trajectories, prediction_horizon, time_step):
    num_agents = initial_states.shape[0]

    fig, ax = plt.subplots()
    for i in range(num_agents):
        ax.plot(trajectories[i][:, 0], trajectories[i][:, 1], label=f'Agent {i+1} Trajectory')
        ax.scatter(initial_states[i, 0], initial_states[i, 1], c='blue', marker='o', label=f'Agent {i+1} Start' if i == 0 else "")
        ax.scatter(reference_states[i, 0], reference_states[i, 1], c='red', marker='x', label=f'Agent {i+1} Goal' if i == 0 else "")

    ax.legend()
    ax.set_title('Trajectories of Agents')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.grid(True)
    plt.show()

# Example usage
initial_states = np.array([[0, 0], [10, 10], [4, 4]])  # Example initial states for three agents in 2D
reference_states = np.array([[10, 10], [0, 0], [7, 7]])  # Example reference states for three agents in 2D
num_agents = initial_states.shape[0]
state_dim = initial_states.shape[1]
prediction_horizon = 10
time_step = 0.5
max_velocity = 1.0
min_distance = 0.5

best_position, best_score = pso_optimize(initial_states, reference_states, num_agents, state_dim, prediction_horizon, time_step, max_velocity, min_distance)

# Extract trajectories from the best position
trajectories = [best_position[i][:state_dim * (prediction_horizon + 1)].reshape((prediction_horizon + 1, state_dim)) for i in range(num_agents)]

plot_trajectories(initial_states, reference_states, trajectories, prediction_horizon, time_step)
