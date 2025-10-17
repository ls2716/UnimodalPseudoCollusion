import numpy as np



def get_reward(action, max_action, noise_level):
    """Sample reward from a trianlge function centered at 0.5 with noise."""
    reward = 1 - np.abs(action - max_action)  # Triangle rewards
    return max(reward + np.random.normal(0, noise_level), 0.0001), reward # Ensure non-negative rewards


def evaluate_algorithm(agent, no_iterations, noise_level, gather_data=False):
    """Evaluate the agent over a number of iterations."""
    # Randomise the best action (uniform between 0.4 and 0.6)
    # This is the center of the triangle
    max_action = np.random.uniform(0.4, 0.6)
    # Set the starting action to the closest to the max_action
    agent.set_starting_action(max_action) 
    # Reset the agent's state
    agent.reset()
    cumulative_reward = 0

    if gather_data:
        rewards = []
        actions = []

    for t in range(no_iterations):
        action_index, action = agent.select_action()
        # Sample a reward based on the action and noise level
        reward, true_reward = get_reward(action, max_action, noise_level)
        agent.update_rewards(action_index, reward)
        cumulative_reward += reward
        if gather_data:
            rewards.append(reward)
            actions.append(action)

    if gather_data:
        return {
            "regret": (no_iterations - cumulative_reward)/no_iterations,  # Return average regret over iterations
            "actions": actions,
            "rewards": rewards,
            "max_action": max_action,
        }
    return (no_iterations - cumulative_reward)/no_iterations  # Return average reward over iterations



# if __name__ == "__main__":
#     # Example usage
#     from sw_eps import SW_EpsilonGreedy


#     no_actions = 201
#     no_neighbors = 5
#     sliding_window = 10
#     epsilon = 1.


#     agent = SW_EpsilonGreedy(
#         epsilon=epsilon,
#         no_actions=no_actions,
#         neighbors=no_neighbors,
#         sliding_window=sliding_window,
#         starting_action=no_actions // 2,
#     )

#     for noise_level in test_parameters["noise_level"]:
#         print(f"Evaluating with noise level: {noise_level}")
#         regret = evaluate_algorithm(agent, no_iterations, noise_level)
#         print(f"Average regret over {no_iterations} iterations: {regret:.4f}")
