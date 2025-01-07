import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def multi_agent_rhc_free_energy(initial_states, reference_states, prior, obs_types, reward_configs = [(0,0,0),(1,1,1)], prediction_horizon=10, time_step=0.1):
    num_agents = initial_states.shape[0]
    state_dim = initial_states.shape[1]
    control_dim = state_dim  # Assuming control dimension matches state dimension for simplicity

    def system_dynamics(x, u):
        return x + u * time_step
    
    def custom_cdist(x1, x2, obs_types, max_distance=1.0):
        distances = []

        for i, t in enumerate(obs_types):
            if t == 's':  # Self-observation
                depth_multiplier = 2.0  # Increase the weight of self-observation
                dist = depth_multiplier * ca.exp(-ca.norm_2(x1[i] - x2.T) / max_distance)
            elif t == 'A':
                dist = ca.exp(-ca.norm_2(x1[i] - x2.T) / max_distance)
            elif t == 'B':
                goal_vectors = x2 - x1[i, :2]
                goal_azimuths = ca.atan2(goal_vectors[:, 1], goal_vectors[:, 0])
                observed_azimuth = ca.atan2(x2[:, 1] - x1[i, 1], x2[:, 0] - x1[i, 0])
                relative_azimuths = ca.atan2(ca.sin(goal_azimuths - observed_azimuth), ca.cos(goal_azimuths - observed_azimuth))
                dist = (1.0 / 8) * ca.exp(-relative_azimuths / ca.pi)
            else:
                dist = ca.MX.zeros(x2.shape[0])
            
            distances.append(dist)
        
        return ca.vertcat(*distances)
    
    def calculate_joint_goal_likelihood(x, x_ref, predict_types, reward_configs, max_distance=1.0):
        num_agents = x.shape[0]
        num_goals = x_ref.shape[0]

        # Calculate distances between agents and goals
        distances = custom_cdist(x, x_ref, predict_types, max_distance)

        # Convert distances to probabilities using softmax
        exp_distances = ca.exp(distances)
        sum_exp_distances = ca.sum1(exp_distances)
        probabilities = exp_distances / ca.repmat(sum_exp_distances.T, distances.shape[0], 1)

        # Initialize joint probabilities as an array of ones with the appropriate shape
        joint_probabilities = np.ones((num_goals,num_agents))
        joint_probabilities_casadi = ca.MX.ones(num_goals, num_agents)

        # Explicitly calculate the joint probabilities
        for config in np.ndindex(*[num_goals] * num_agents):
            prob = 1.0
            for i in range(num_agents):
                prob *= probabilities[i, config[i]]
            joint_probabilities_casadi[config] = prob

        # Only return the specified configurations
        likelihood = ca.vertcat(*[joint_probabilities_casadi[config] for config in reward_configs])

        # Normalize the joint probabilities
        likelihood /= ca.sum1(likelihood)

        return likelihood
    
    def entropy(x):
        return -ca.sum1(x * ca.log(x))

    def free_energy(x, x_ref, prior, obs_types, reward_configs):
        belief = calculate_joint_goal_likelihood(x, x_ref, obs_types,reward_configs)
        return entropy(belief)


    opti = ca.Opti()
    X = [[opti.variable(state_dim) for _ in range(prediction_horizon + 1)] for _ in range(num_agents)]
    U = [[opti.variable(state_dim) for _ in range(prediction_horizon)] for _ in range(num_agents)]

    X_ref = opti.parameter(state_dim, num_agents)
    U_ref = np.zeros((state_dim, num_agents))

    total_cost = 0
    max_velocity = 1
    min_distance = 0.5

    for i in range(num_agents):

        opti.subject_to(X[i][0] == initial_states[i])
        for k in range(prediction_horizon):
            opti.subject_to(X[i][k + 1] == system_dynamics(X[i][k], U[i][k]))
            for dim in range(state_dim):
                opti.subject_to(U[i][k][dim] <= max_velocity)
                opti.subject_to(U[i][k][dim] >= -max_velocity)

            prediction_error = free_energy(X[i][k], X_ref[:, i], prior, obs_types, reward_configs)
            total_cost += prediction_error  # + 0.01 * control_effort

    for k in range(prediction_horizon):
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                distance = ca.norm_2(X[i][k] - X[j][k])
                collision_avoidance_cost = ca.fmax(0, min_distance - distance) ** 2
                total_cost += 1e6 * collision_avoidance_cost  # High penalty for collisions
                opti.subject_to(distance >= min_distance)

    opti.minimize(total_cost)

    # Initialize variables
    for i in range(num_agents):
        for k in range(prediction_horizon + 1):
            opti.set_initial(X[i][k], initial_states[i])
        for k in range(prediction_horizon):
            opti.set_initial(U[i][k], 0)

    p_opts = {"verbose": False}
    s_opts = {"max_iter": 500, "print_level": 0}  # Reduced maximum iterations for faster convergence
    opti.solver('ipopt', p_opts, s_opts)

    opti.set_value(X_ref, reference_states.T)
    
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(f"Solver failed: {e}")
        return None, None

    optimal_u = np.zeros((num_agents, control_dim))
    for i in range(num_agents):
        for dim in range(control_dim):
            optimal_u[i][dim] = sol.value(U[i][0][dim])

    trajectories = np.zeros((state_dim, prediction_horizon + 1, num_agents))
    for i in range(num_agents):
        for k in range(prediction_horizon + 1):
            for dim in range(state_dim):
                trajectories[dim, k, i] = sol.value(X[i][k][dim])

    return optimal_u, trajectories

def plot_trajectories(initial_states, reference_states, optimal_control_inputs, trajectories, prediction_horizon, time_step):
    num_agents = initial_states.shape[0]

    fig, ax = plt.subplots()
    for i in range(num_agents):
        ax.plot(trajectories[0, :, i], trajectories[1, :, i], label=f'Agent {i+1} Trajectory')
        ax.scatter(initial_states[i, 0], initial_states[i, 1], c='blue', marker='o', label=f'Agent {i+1} Start' if i == 0 else "")
        ax.scatter(reference_states[i, 0], reference_states[i, 1], c='red', marker='x', label=f'Agent {i+1} Goal' if i == 0 else "")

    ax.legend()
    ax.set_title('Trajectories of Agents')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.grid(True)
    plt.show()

# Example usage:
initial_states = np.array([[10, 0], [0, 10]])  # Example initial states for two agents in 2D
reference_states = np.array([[10, 10], [0, 0]])  # Example reference states for two agents in 2D
prior = np.repeat(1.0 / reference_states.shape[0], reference_states.shape[0])  # Uniform prior
obs_types = ['s', 's']  # Self-observation for all agents
prediction_horizon = 15
time_step = 0.5

optimal_control_inputs, trajectories = multi_agent_rhc_free_energy(initial_states, reference_states, prior, obs_types, [(0,0),(1,1)],prediction_horizon, time_step)
if trajectories is not None:
    plot_trajectories(initial_states, reference_states, optimal_control_inputs, trajectories, prediction_horizon, time_step)
