import numpy as np


class SW_UTS:
    def __init__(
        self,
        no_actions=11,
        no_neighbors=2,
        starting_action=0,
        sliding_window=100,
        std=0.1,
    ):
        self.no_actions = int(no_actions)
        self.no_neighbors = int(no_neighbors)
        self.sliding_window = int(sliding_window)
        self.variance = std**2

        self.rewards = np.zeros((self.sliding_window))
        self.was_chosen = (np.zeros((self.sliding_window)) - 1).astype(
            int
        )  # Initialize with -1 to indicate no action was chosen

        self.average_rewards = np.zeros(self.no_actions)
        self.counts = np.zeros(
            self.no_actions
        )  # Count of how many times each action was chosen
        self.action_space = np.linspace(0, 1, self.no_actions, endpoint=True)

        self._buffer_idx = 0  # Circular buffer index
        self.set_starting_action(starting_action)
        self.incumbent = self.starting_action_index

    def select_action(self):
        # Get the neighbors of the incumbent action
        left = max(0, self.incumbent - self.no_neighbors)
        right = min(self.no_actions, self.incumbent + self.no_neighbors + 1)
        variances = self.variance / (self.counts[left:right] + 1e-5)
        values_neighbors = np.random.normal(
            self.average_rewards[left:right], np.sqrt(variances)
        ) + np.random.normal(0, 0.0001, size=(right - left))

        action_index = (
            np.argmax(values_neighbors) + left
        )  # Adjust index to the full action space
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
                    self.average_rewards[outgoing_action_index] * count
                    - outgoing_reward
                ) / (count - 1)
            else:
                self.average_rewards[outgoing_action_index] = 0
            # Decrease the count for the outgoing action
            self.counts[outgoing_action_index] -= 1

        # Update the rewards and was_chosen matrices
        self.rewards[self._buffer_idx] = reward
        self.was_chosen[self._buffer_idx] = action_index
        self.average_rewards[action_index] = (
            self.average_rewards[action_index] * self.counts[action_index] + reward
        ) / (self.counts[action_index] + 1)
        self.counts[action_index] += 1

        # Update the circular buffer index
        self._buffer_idx = (self._buffer_idx + 1) % self.sliding_window

        self.incumbent = np.argmax(self.average_rewards)

    def reset(self):
        """Reset the agent's state."""
        self.average_rewards = np.zeros(self.no_actions)
        self.counts = np.zeros(self.no_actions)  # Reset counts
        self.rewards = np.zeros((self.sliding_window))
        self.was_chosen = np.zeros((self.sliding_window)).astype(int) - 1  #
        self._buffer_idx = 0
        self.incumbent = self.starting_action_index

    def set_starting_action(self, action):
        """Set the starting action for the agent."""
        if 0 > action or action > 1:
            raise ValueError("Action must be in the range [0, 1]")
        self.starting_action_index = np.argmin(np.abs(action - self.action_space))
