import numpy as np



def get_reward(action, noise_level, timestep, no_iterations):
    """Sample reward from a trianlge function centered at 0.5 with noise."""
    max_action = 0.1 + 0.8 * timestep / no_iterations  # Non-stationary max action
    reward = 1 - np.abs(action - max_action)   # Triangle rewards
    return max(reward + np.random.normal(0, noise_level), 0.001), reward # Ensure non-negative rewards


def evaluate_algorithm(agent, no_iterations, noise_level, gather_data=False):
    """Evaluate the agent over a number of iterations."""
    # Set the starting action
    agent.set_starting_action(0.1)  # Centered at 0.1
    # Reset the agent's state
    agent.reset()

    cumulative_reward = 0
    cumulative_true_reward = 0
    if gather_data:
        rewards = []
        actions = []

    for t in range(no_iterations):
        action_index, action = agent.select_action()
        # Sample a reward based on the action and noise level
        reward, true_reward = get_reward(action, noise_level, t, no_iterations)
        agent.update_rewards(action_index, reward)
        cumulative_reward += reward
        cumulative_true_reward += true_reward
        if gather_data:
            rewards.append(reward)
            actions.append(action)

    if gather_data:
        return {
            "regret": (no_iterations - cumulative_reward)/no_iterations,  # Return average regret over iterations
            "actions": actions,
            "rewards": rewards,
        }
    

    return (no_iterations - cumulative_reward)/no_iterations  # Return average reward over iterations


