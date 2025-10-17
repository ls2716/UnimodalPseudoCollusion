import numpy as np


def get_probs(prices, a, c, mu, a_0):
    """
    Calculate the probabilities of choosing each action based on a logistic function.

    Args:
        prices (np.ndarray): Array of prices (shape: [batch, n_agents]).
        a (float): Parameter for the logistic function.
        c (float): Cost parameter (not used directly here).
        mu (float): Smoothing parameter for the logistic function.
        a_0 (float): Baseline action parameter.

    Returns:
        np.ndarray: Probabilities for each action (shape: [batch, n_agents]).
    """
    exps = np.exp((a - prices) / mu)
    probs = exps / (np.sum(exps, axis=1, keepdims=True) + np.exp(a_0 / mu))
    return probs


def get_rewards(prices, a, c, mu, a_0, covariance_matrix):
    """
    Calculate the rewards for each agent, using the logistic function and adding multivariate Gaussian noise.

    Args:
        prices (np.ndarray): Array of prices (shape: [batch, n_agents]).
        a (float): Parameter for the logistic function.
        c (float): Cost parameter.
        mu (float): Smoothing parameter for the logistic function.
        a_0 (float): Baseline action parameter.
        covariance_matrix (np.ndarray): Covariance matrix for the noise (shape: [n_agents, n_agents]).

    Returns:
        np.ndarray: Rewards for each agent (shape: [batch, n_agents]).
    """
    probs = get_probs(prices, a, c, mu, a_0)
    random_noise = np.random.multivariate_normal(
        mean=np.zeros(prices.shape[1]), cov=covariance_matrix, size=prices.shape[0]
    )
    rewards = (prices - c) * probs + random_noise
    return rewards


def find_nash(N, a, c, mu, a_0):
    """
    Find the Nash equilibrium price for a symmetric game.

    Args:
        N (int): Number of agents.
        a (float): Parameter for the logistic function.
        c (float): Cost parameter.
        mu (float): Smoothing parameter for the logistic function.
        a_0 (float): Baseline action parameter.

    Returns:
        float: Nash equilibrium price (for the first agent, symmetric case).
    """
    print("Finding Nash equilibrium...")
    p = np.zeros(shape=(1, N)) + c + np.random.rand(1, N) * 0.01
    for iteration in range(100):
        p_new = (c + mu / (1 - get_probs(p, a, c, mu, a_0))).reshape(1, N)
        p_new = p + (p_new - p) * 0.2  # Smooth the update
        if np.all(np.abs(p_new - p) < 1e-5):
            break
        p = p_new
    if iteration == 19:
        print("Warning: find_nash did not converge within 20 iterations.")
    # Find the nash reward
    p = p.reshape(1, N)
    reward = (p - c) * get_probs(p, a, c, mu, a_0)

    return p[0, 0], reward[0, 0]


def run_episode(
    agents,
    iterations,
    a,
    c,
    mu,
    a_0,
    variance=0.0,
    correlation=0,
    nash_price=None,
    bounds=None,
):
    """
    Run a single episode for a set of agents, simulating their pricing and reward process.

    Args:
        agents (list): List of agent objects, each with select_action, set_starting_action, reset, and update_rewards methods.
        iterations (int): Number of time steps in the episode.
        a (float): Parameter for the logistic function.
        c (float): Cost parameter.
        mu (float): Smoothing parameter for the logistic function.
        a_0 (float): Baseline action parameter.
        variance (float, optional): Variance for the reward noise. Default is 0.
        correlation (float, optional): Correlation for the reward noise. Default is 0.
        nash_price (float, optional): Nash equilibrium price. If None, it will be computed.
        bounds (list of tuple, optional): List of (min, max) tuples for each agent's price range. If None, actions are mapped to [c, c+1].

    Returns:
        tuple: (prices, rewards) where both are np.ndarray of shape [iterations, n_agents].
    """
    n_agents = len(agents)
    prices = np.zeros((iterations, n_agents))
    rewards = np.zeros((iterations, n_agents))

    if nash_price is None:
        nash_price, _ = find_nash(n_agents, a, c, mu, a_0)

    nash_margin = nash_price - c
    nash_prices = np.ones(shape=(1, n_agents)) * nash_price
    nash_rewards = (nash_prices - c) * get_probs(nash_prices, a, c, mu, a_0)
    nash_reward = nash_rewards[0, 0]

    if bounds is None:
        bounds = (nash_margin-0.5, nash_margin+0.5)   

    # Create covariance matrix for the noise
    covariance_matrix = np.full((n_agents, n_agents), correlation * variance)
    np.fill_diagonal(covariance_matrix, variance)

    # Initialize agents to Nash equilibrium margin
    for agent in agents:
        agent.set_starting_action(0.5)
        agent.reset()

    for t in range(iterations):
        # Each agent selects an action (price index)
        action_indexes = []
        actions = []
        for agent in agents:
            action_index, action = agent.select_action()
            action_indexes.append(action_index)
            actions.append(
                action * (bounds[1]-bounds[0]) + bounds[0]
            )  # Scale action to [bounds[0], bounds[1]]

        # Set prices as the scaled actions plus cost c
        prices[t] = c + np.array(actions)

        # Calculate rewards for this time step
        rewards[t] = get_rewards(
            prices[t].reshape(1, -1), a, c, mu, a_0, covariance_matrix=covariance_matrix
        )

        # Update each agent with their reward
        for i, agent in enumerate(agents):
            agent.update_rewards(action_indexes[i], rewards[t, i])

    return prices, rewards, nash_price, nash_reward

def run_multiple_episodes(
    no_sims,
    agents,
    iterations,
    a,
    c,
    mu,
    a_0,
    variance=0.0,
    correlation=0,
    nash_price=None,
    bounds=None,
    analyse_each=True,
):
    """
    Run multiple episodes for a set of agents, simulating their pricing and reward process.

    Args:
        agents (list): List of agent objects, each with select_action, set_starting_action, reset, and update_rewards methods.
        iterations (int): Number of time steps in each episode.
        a (float): Parameter for the logistic function.
        c (float): Cost parameter.
        mu (float): Smoothing parameter for the logistic function.
        a_0 (float): Baseline action parameter.
        variance (float, optional): Variance for the reward noise. Default is 0.
        correlation (float, optional): Correlation for the reward noise. Default is 0.
        nash_price (float, optional): Nash equilibrium price. If None, it will be computed.
        bounds (list of tuple, optional): List of (min, max) tuples for each agent's price range.

    Returns:
        list: List of tuples containing (prices, rewards, nash_price, nash_reward) for each episode.
    """
    results = []
    for episode in range(no_sims):  # Run no_sims episodes
        prices, rewards, nash_price, nash_reward = run_episode(
            agents,
            iterations,
            a,
            c,
            mu,
            a_0,
            variance,
            correlation,
            nash_price,
            bounds=None,
        )
            
        results.append((prices, rewards, nash_price, nash_reward))
    return results


def analyse_episode(prices, rewards, nash_price, nash_reward):
    """
    Analyse the results of an episode, calculating average prices and rewards.

    Args:
        prices (np.ndarray): Array of prices (shape: [iterations, n_agents]).
        rewards (np.ndarray): Array of rewards (shape: [iterations, n_agents]).
        nash_price (float): Nash equilibrium price.
        nash_reward (float): Reward at Nash equilibrium.
        c (float): Cost parameter.

    Returns:
        dict: Dictionary containing average prices, average rewards, and Nash metrics.
    """
    avg_prices = np.mean(prices, axis=0)
    avg_rewards = np.mean(rewards, axis=0)
    std_prices = np.std(prices, axis=0)

    analysis = {
        "average_prices": avg_prices,
        "std_prices": std_prices,
        "average_rewards": avg_rewards,
        "nash_price": nash_price,
        "nash_reward": nash_reward,
    }

    # Compute correlation between margins of different agents
    if prices.shape[1] > 1:
        correlation_matrix = np.corrcoef(prices.T)
        analysis["prices_correlation"] = correlation_matrix
    else:
        analysis["prices_correlation"] = np.array([[1.0]])

    # Compute the standard deviation of prices for each agent
    std_prices = np.std(prices, axis=0)
    analysis["std_prices"] = std_prices

    return analysis





def analyse_aggregate(results_list):
    """
    Analyse aggregate results from multiple episodes.

    Args:
        results_list (list): List of tuples containing (prices, rewards, nash_price, nash_reward) for each episode.
        each nash price and nash_reward is the same across episodes.

    Returns:
        dict: Dictionary containing aggregated analysis across all episodes.
    """
    avg_prices = []
    avg_rewards = []
    std_prices = []
    prices_correlation = []
    nash_price = results_list[0][2]
    for prices, rewards, nash_price, nash_reward in results_list:
        analysis = analyse_episode(prices, rewards, nash_price, nash_reward)
        avg_prices.append(analysis["average_prices"])
        avg_rewards.append(analysis["average_rewards"])
        std_prices.append(analysis["std_prices"])
        prices_correlation.append(analysis["prices_correlation"])

    avg_price_all = np.mean(avg_prices, axis=0)
    std_price_all = np.mean(std_prices, axis=0)


    aggregate_analysis = {
        "average_prices": avg_prices,
        "std_prices": std_prices,
        "overall_average_prices": avg_price_all,
        "overall_std_prices": std_price_all,
        "nash_price": nash_price,
        "prices_correlation": prices_correlation,
    }

    return aggregate_analysis


def aggregate_prices(results_list):
    """
    Aggregate prices from multiple episodes.

    Args:
        results_list (list): List of tuples containing (prices, rewards, nash_price, nash_reward) for each episode.

    Returns:
        np.ndarray: Aggregated prices across all episodes (shape: [total_iterations, n_agents]).
    """
    all_prices = np.vstack([result[0] for result in results_list])
    return all_prices