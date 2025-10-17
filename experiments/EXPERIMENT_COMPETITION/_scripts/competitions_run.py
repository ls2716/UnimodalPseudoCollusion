import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from cases_definition import cases
from agent_definition import get_agent
from tools.competition import run_episode, find_nash

PARAMS = {
    "a": 1.0,
    "c": 5.0,
    "mu": 0.25,
    "a_0": -5.0,
    "stds": [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
    "correlations": [ 0.0,],
}
MIN_NO_SIMS = 20
MAX_NO_SIMS = 100
MIN_MARGIN_ACCURACY = (
    0.005  # Maximum ratio of std to mean for the margin of error to be acceptable
)
ITERATIONS = 2000
ITERATIONS_PLOT = 2000
RECREATE_RESULTS = False  # Whether to recreate results even if they exist


def run_noise_correlation_case(name, std, corr):
    print("--------------------------------------------------")
    print(f"Running case '{name}' with std={std}, corr={corr}")
    # Load case definition
    if name not in cases:
        raise ValueError(f"Case '{name}' not found in cases definition.")
    case = cases[name]
    agents_names = case["agents"]
    n_agents = len(agents_names)

    # Get agents
    agents = [get_agent(agent_name) for agent_name in agents_names]

    # Find the NASH price for the given parameters
    nash_price, nash_reward = find_nash(
        N=n_agents, a=PARAMS["a"], c=PARAMS["c"], mu=PARAMS["mu"], a_0=PARAMS["a_0"]
    )
    print(f"Nash price: {nash_price}, Nash reward: {nash_reward}")

    # Check if results already exist
    output_dir = f"../competition_results/{name}/std_{std}_corr_{corr}/"
    output_filepath = os.path.join(output_dir, "results.pkl")
    if os.path.exists(output_filepath) and not RECREATE_RESULTS:
        print(
            f"Results for case '{name}' with std={std}, corr={corr} already exist. Skipping..."
        )
        return

    it = 0
    avg_prices = []
    all_prices = []
    all_rewards = []

    while it < MAX_NO_SIMS:
        it += 1
        episode_prices, episode_rewards, _, _ = run_episode(
            agents=agents,
            iterations=ITERATIONS,
            a=PARAMS["a"],
            c=PARAMS["c"],
            mu=PARAMS["mu"],
            a_0=PARAMS["a_0"],
            variance=std**2,
            correlation=corr,
            nash_price=nash_price,
        )
        all_prices.append(episode_prices)
        all_rewards.append(episode_rewards)
        avg_prices.append(episode_prices.mean(axis=0))

        if it >= MIN_NO_SIMS:
            avg_prices_array = np.array(avg_prices)
            avg_margin_array = avg_prices_array - PARAMS["c"]
            global_avg_price = avg_prices_array.mean(axis=0)
            global_avg_price_std = avg_prices_array.std(axis=0) / np.sqrt(it)
            global_avg_margin = avg_margin_array.mean(axis=0)
            global_avg_margin_std = avg_margin_array.std(axis=0) / np.sqrt(it)
            # Use first agent to check the margin of error
            ratio = global_avg_margin_std[0] / global_avg_margin[0]
            if ratio < MIN_MARGIN_ACCURACY:
                print(
                    f"Acceptable margin of error reached after {it} simulations (ratio={ratio:.4f})."
                )
                break

    print(f"Completed {it} simulations.")
    print(f"Average prices: {global_avg_price}")
    print(f"Average prices std: {global_avg_price_std}")
    print(f"Average margins: {global_avg_margin}")
    print(f"Average margins std: {global_avg_margin_std}")
    # Print average margin increases with stds of margin increase in percentage
    nash_margin = nash_price - PARAMS["c"]
    global_avg_margin_increase = (global_avg_margin / nash_margin - 1) * 100
    global_avg_margin_std_percentage = (global_avg_margin_std / nash_margin) * 100
    print(f"Average margin increase over Nash: {global_avg_margin_increase}%")
    print(f"Average margin std (percentage points): {global_avg_margin_std_percentage}%")

    all_prices_np = np.array(all_prices)
    all_rewards_np = np.array(all_rewards)
    # Save the results to a file - use pickle
    os.makedirs(output_dir, exist_ok=True)
    with open(output_filepath, "wb") as f:
        pickle.dump((all_prices_np, all_rewards_np, nash_price, nash_reward), f)

    # Plot a sample episode
    sample_prices = all_prices_np[0]
    plt.figure(figsize=(10, 6))
    for i in range(n_agents):
        plt.plot(sample_prices[:, i], label=f"Agent {i + 1} ({agents_names[i]})")
    plt.axhline(y=nash_price, color="r", linestyle="--", label="Nash Price")
    plt.title(f"Prices over Time - Case '{name}' (std={std}, corr={corr})")
    plt.ylim([nash_price-0.3, nash_price+0.3])
    plt.xlabel("Iteration")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plot_filepath = os.path.join(output_dir, "price_plot.png")
    plt.savefig(plot_filepath)
    plt.close()


def run_case(name):
    # Load case definition
    if name not in cases:
        raise ValueError(f"Case '{name}' not found in cases definition.")

    # Run multiple episodes for each standard deviation and correlation
    for std in PARAMS["stds"]:
        for corr in PARAMS["correlations"]:
            run_noise_correlation_case(name, std, corr)


if __name__ == "__main__":
    for case_name in cases.keys():
        run_case(case_name)
