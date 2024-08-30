import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def multi_agent_rhc_free_energy(initial_states, reference_states, prediction_horizon=10, time_step=0.1):
    num_agents = initial_states.shape[0]
    state_dim = initial_states.shape[1]
    control_dim = state_dim  # Assuming control dimension matches state dimension for simplicity

    def system_dynamics(x, u):
        return x + u * time_step

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
            prediction_error = ca.sumsqr(X[i][k] - X_ref[:, i])
            control_effort = ca.sumsqr(U[i][k] - U_ref[:, i])
            total_cost += prediction_error  # + 0.01 * control_effort
            for dim in range(state_dim):
                opti.subject_to(U[i][k][dim] <= max_velocity)
                opti.subject_to(U[i][k][dim] >= -max_velocity)

    for k in range(prediction_horizon):
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                distance = ca.sqrt(ca.sumsqr(X[i][k] - X[j][k]))
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
initial_states = np.array([[0, 0], [10, 10], [4,4]])  # Example initial states for three agents in 2D
reference_states = np.array([[10, 10], [0, 0], [7,7]])  # Example reference states for three agents in 2D
prediction_horizon = 15
time_step = 0.5

optimal_control_inputs, trajectories = multi_agent_rhc_free_energy(initial_states, reference_states, prediction_horizon, time_step)
if trajectories is not None:
    plot_trajectories(initial_states, reference_states, optimal_control_inputs, trajectories, prediction_horizon, time_step)
