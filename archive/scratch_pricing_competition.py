import numpy as np
import matplotlib.pyplot as plt


def get_probs(prices, a, c, mu, a_0):
    """
    Calculate the probabilities of choosing each action based on the logistic function.
    """
    exps = np.exp((a - prices) / mu)
    probs = exps / (np.sum(exps, axis=1, keepdims=True) + np.exp(a_0 / mu))
    # print(f"Calculated probabilities: {probs}")
    return probs


def get_rewards(prices, a, c, mu, a_0, covariance_matrix):
    """
    Calculate the rewards according to the logistic function with added noise.
    """
    probs = get_probs(prices, a, c, mu, a_0)
    random_noise = np.random.multivariate_normal(
        mean=np.zeros(prices.shape[1]), cov=covariance_matrix, size=prices.shape[0]
    )
    rewards = (prices - c) * probs + random_noise
    return rewards


def find_nash(N, a, c, mu, a_0):
    """
    Find the Nash equilibrium prices - symmetric case.
    """
    p = np.ones(shape=(1, N)) + c + np.random.rand(1, N) * 0.01
    for _ in range(20):
        p_new = (c + mu / (1 - get_probs(p, a, c, mu, a_0))).reshape(1, N)
        p_new = p + (p_new - p) * 0.25  # Smooth the update
        if np.all(np.abs(p_new - p) < 1e-6):
            break
        p = p_new
    print(f"Converged to Nash equilibrium after {_} iterations.")
    return p[0, 0]


def run_episode(
    agents, iterations, a, c, mu, a_0, variance=0.0, correlation=0, nash_price=None
):
    """
    Run a single episode for the agents and return the total rewards.
    """
    n_agents = len(agents)
    prices = np.zeros((iterations, n_agents))
    rewards = np.zeros((iterations, n_agents))

    if nash_price is None:
        nash_price = find_nash(n_agents, a, c, mu, a_0)

    nash_margin = nash_price - c

    covariance_matrix = np.full((n_agents, n_agents), correlation * variance)
    np.fill_diagonal(covariance_matrix, variance)

    # Set the starting prices of the agents to the Nash equilibrium margins
    for agent in agents:
        agent.set_starting_action(nash_margin)
        agent.reset()

    for t in range(iterations):
        action_indexes = [agent.select_action()[0] for agent in agents]
        margins = [
            action_index / (agent.no_actions - 1)
            for action_index, agent in zip(action_indexes, agents)
        ]
        prices[t] = c + np.array(margins)
        rewards[t] = get_rewards(
            prices[t].reshape(1, -1), a, c, mu, a_0, covariance_matrix=covariance_matrix
        )

        for i, agent in enumerate(agents):
            agent.update_rewards(action_indexes[i], rewards[t, i])

    return prices, rewards


if __name__ == "__main__":
    # Finding the Nash equilibrium for a symmetric case
    N = 2
    a = 1
    c = 5
    mu = 0.25
    a_0 = -1 - c

    p = find_nash(N, a, c, mu, a_0)
    print(f"Nash Equilibrium Prices: {p}")
    # Compute the reward at the Nash equilibrium
    rewards = get_rewards(
        np.array([[p] * N]), a, c, mu, a_0, covariance_matrix=np.eye(N) * 0.00
    )
    print(f"Rewards at Nash Equilibrium: {rewards}")

    # Plot the rewards with varying one agent price
    price_range = np.linspace(c, c + 1, 100)
    prices = np.zeros((len(price_range), N))
    prices[:, 0] = price_range
    prices[:, 1:] = p

    rewards = get_rewards(prices, a, c, mu, a_0, covariance_matrix=np.eye(N) * 0.00)
    plt.plot(price_range, rewards[:, 0], label="Agent 1 Rewards")
    # Mark the Nash equilibrium price
    plt.axvline(x=p, color="r", linestyle="--", label="Nash Equilibrium Price")
    plt.xlabel("Agent 1 Price")
    plt.ylabel("Rewards")
    plt.title("Rewards vs Agent 1 Price")
    plt.legend()
    plt.show()

    # Example usage of run_episode
    from tools.bandits.sw_eps import SW_EpsilonGreedy

    agents = [
        SW_EpsilonGreedy(sliding_window=25, epsilon=0.3, no_actions=100, no_neighbors=7)
        for _ in range(N)
    ]

    # noise correlation = 0.5
    # Create a covariance matrix for the noise
    std = 0.01
    variance = std**2
    correlation = 1.

    iterations = 20000

    from tools.competition import run_episode, analyse

    prices, rewards, nash_price, nash_reward = run_episode(
        agents,
        iterations,
        a,
        c,
        mu,
        a_0,
        variance=variance,
        correlation=correlation,
        nash_price=p,
    )

    # Run the analysis
    analysis_results = analyse(prices, rewards, nash_price, nash_reward, c)
    # Pretty print the analysis results
    for key, value in analysis_results.items():
        print(f"{key}: {value}")

    # Plot the rewards and prices over time
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(prices)
    plt.title("Prices Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Price")
    plt.legend([f"Agent {i + 1}" for i in range(N)])
    plt.subplot(2, 1, 2)
    plt.plot(rewards)
    plt.title("Rewards Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.legend([f"Agent {i + 1}" for i in range(N)])
    plt.tight_layout()
    plt.savefig("epsilong_greedy_prices_rewards.png")
    plt.show()

    # Plot the 2D histogram of prices for the first two agents
    plt.figure(figsize=(8, 6))
    plt.hist2d(prices[:, 0], prices[:, 1], bins=40, cmap="Blues")
    plt.colorbar(label="Frequency")
    plt.title("2D Histogram of Prices for Agent 1 and Agent 2")
    plt.xlabel("Agent 1 Price")
    plt.ylabel("Agent 2 Price")
    plt.grid()
    plt.tight_layout()
    plt.savefig("epsilong_greedy_2d_histogram_prices.png")
    plt.show()
