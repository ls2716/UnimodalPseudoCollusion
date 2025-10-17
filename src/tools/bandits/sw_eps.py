"""Implements the SW-epsilong greedy algorithm with local exploration."""



import numpy as np



class SW_EpsilonGreedy:
    def __init__(self, epsilon=0.1, no_actions=6, no_neighbors=2, starting_action=0, sliding_window=10):
        self.no_actions = int(no_actions)
        self.no_neighbors = int(no_neighbors)
        self.epsilon = epsilon
        self.sliding_window = int(sliding_window)

        self.rewards = np.zeros((self.sliding_window))
        self.was_chosen = (np.zeros((self.sliding_window))-1).astype(int)  # Initialize with -1 to indicate no action was chosen

        self.average_rewards = np.zeros(self.no_actions)
        self.counts = np.zeros(self.no_actions)  # Count of how many times each action was chosen
        self.action_space = np.linspace(0, 1, self.no_actions, endpoint=True)

        self._buffer_idx = 0  # Circular buffer index
        self.set_starting_action(starting_action)
        self.incumbent = self.starting_action_index

    def select_action(self):
        if np.random.rand() < self.epsilon:
            # Explore: select a random neighbor (excluding incumbent)
            left = max(0, self.incumbent - self.no_neighbors)
            right = min(self.no_actions, self.incumbent + self.no_neighbors + 1)
            action_index = np.random.randint(left, right)
        else:
            # Exploit: select the best action
            action_index = self.incumbent

        action = self.action_space[action_index]
        return action_index, action
        

    def update_rewards(self, action_index, reward):

        # Get the index of the action that was chosen that goes out of the sliding window
        outgoing_action_index = self.was_chosen[self._buffer_idx]
        outgoing_reward = self.rewards[self._buffer_idx]

        # Update the average reward for the action that goes out of the sliding window
        if outgoing_action_index != -1:  # If an action was chosen
            count = self.counts[outgoing_action_index]
            if count > 1:
                self.average_rewards[outgoing_action_index] = (
                    self.average_rewards[outgoing_action_index] * count - outgoing_reward) / (count - 1)
            else:
                self.average_rewards[outgoing_action_index] = 0
            # Decrease the count for the outgoing action
            self.counts[outgoing_action_index] -= 1

        # Update the rewards and was_chosen matrices
        self.rewards[self._buffer_idx] = reward
        self.was_chosen[self._buffer_idx] = action_index
        self.average_rewards[action_index] = (
            self.average_rewards[action_index] * self.counts[action_index] + reward) / (self.counts[action_index] + 1)
        self.counts[action_index] += 1

        # Update the circular buffer index
        self._buffer_idx = (self._buffer_idx + 1) % self.sliding_window

        self.incumbent = np.argmax(self.average_rewards)

    
    def set_starting_action(self, action):
        """Set the starting action for the agent."""
        if 0>action or action>1:
            raise ValueError("Action must be in the range [0, 1]")
        self.starting_action_index = np.argmin(np.abs(action - self.action_space))
        

    def reset(self):
        """Reset the agent's state."""
        self.average_rewards = np.zeros(self.no_actions)
        self.counts = np.zeros(self.no_actions)  # Reset counts
        self.rewards = np.zeros((self.sliding_window))
        self.was_chosen = np.zeros((self.sliding_window)).astype(int) - 1  #
        self._buffer_idx = 0
        self.incumbent = self.starting_action_index


if __name__ == "__main__":
    # Example usage
    agent = SW_EpsilonGreedy(epsilon=0.8, no_actions=11, starting_action=0.5)

    for i in range(20):
        action_index, action = agent.select_action()
        agent.update_rewards(action_index, i)  # Simulate receiving a reward of 1.0
        # _ = input("Press Enter to continue...")  # Pause after each iteration to observe changes