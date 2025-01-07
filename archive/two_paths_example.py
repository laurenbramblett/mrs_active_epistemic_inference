import numpy as np
import matplotlib.pyplot as plt

# Initialize random seed for reproducibility
np.random.seed(42)

class ParticleFilter:
    def __init__(self, initial_position, N=100):
        self.N = N  # Number of particles
        self.particles = np.random.normal(initial_position, [1, 1], (N, 2))
        self.weights = np.ones(N) / N

    def predict(self, move):
        """Predicts the next state of the particles based on the move."""
        self.particles += np.random.normal(move, [0.5, 0.5], (self.N, 2))

    def update(self, measurement):
        """Updates the weights of the particles based on the measurement."""
        dists = np.linalg.norm(self.particles - measurement, axis=1)
        self.weights = np.exp(-dists**2 / 1.0)  # Assuming R=1 for simplicity
        self.weights += 1.e-300  # Avoid zeros
        self.weights /= sum(self.weights)

    def estimate(self):
        """Estimates the current position based on particles and weights."""
        return np.average(self.particles, weights=self.weights, axis=0)

    def resample(self):
        """Resamples particles based on their weights."""
        indices = np.random.choice(self.N, self.N, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.N) / self.N

def navigate_and_decide(pf, current_position, target_position, steps=5):
    moves = {'A': np.array([1, 0]), 'B': np.array([0, 1])}  # Simplified moves: right (A) or up (B)
    path_taken = []

    for step in range(steps):
        vfe_scores = {}
        for path, move in moves.items():
            pf_temp = ParticleFilter(pf.particles.mean(axis=0), N=pf.N)
            pf_temp.particles = np.copy(pf.particles)  # Copy current particle distribution
            pf_temp.weights = np.copy(pf.weights)

            # Predict and measure for both paths
            pf_temp.predict(move)
            measurement = current_position + move + np.random.normal(0, 0.5, 2)
            pf_temp.update(measurement)
            pf_temp.resample()

            # Calculate VFE (using variance as a proxy)
            vfe_scores[path] = np.var(pf_temp.particles, axis=0).mean()

        # Choose path with lower VFE
        chosen_path = min(vfe_scores, key=vfe_scores.get)
        path_taken.append(chosen_path)

        # Move and update PF based on chosen path
        move = moves[chosen_path]
        pf.predict(move)
        current_position += move
        measurement = current_position + np.random.normal(0, 0.5, 2)
        pf.update(measurement)
        pf.resample()

        # Visualization after decision
        plt.scatter(pf.particles[:, 0], pf.particles[:, 1], label=f'Step {step+1}: Path {chosen_path}', alpha=0.5)

    plt.scatter(target_position[0], target_position[1], marker='x', color='red', label='Target', s=200)
    plt.title('Rover Navigation and Particle Distribution')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.show()

    return path_taken

# Initial setup
initial_position = np.array([0, 0])
target_position = np.array([4, 4])  # Target location the rover aims to reach
pf = ParticleFilter(initial_position)

# Start navigation and decision-making
path_taken = navigate_and_decide(pf, initial_position, target_position, steps=8)
print("Path Taken:", " -> ".join(path_taken))
