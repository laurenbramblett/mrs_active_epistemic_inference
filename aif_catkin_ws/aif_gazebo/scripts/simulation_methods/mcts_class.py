import numpy as np

class MCTSNode:
    def __init__(self, position, prior, num_actions, observations = None, velocity=None, heading=None, parent=None):
        self.position = position
        self.prior = prior
        self.velocity = velocity
        self.heading = heading
        self.observations = observations
        self.parent = parent
        self.num_actions = num_actions
        self.children = []
        self.visits = 1.0
        self.value = 1.0

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, value):
        self.visits += 1
        self.value = np.min((self.value,value))

    def is_fully_expanded(self):
        return len(self.children) >= self.num_actions

    def best_child(self, exploration_weight=2.0):
        choices_weights = [
            -(child.value / (child.visits+1)) + exploration_weight * np.sqrt((2 * np.log(self.visits) / (child.visits+1e-16)))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]
    def least_visited_child(self):
        return min(self.children, key=lambda child: child.visits) if self.children else None
