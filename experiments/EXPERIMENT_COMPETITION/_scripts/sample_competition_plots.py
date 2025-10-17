import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from competitions_run import PARAMS
from tools.competition import find_nash


def load_results(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    with open(filepath, "rb") as f:
        results = pickle.load(f)
    return results


cases_dict = {
    "eps_vs_eps": {"agents": ["epsilon_greedy", "epsilon_greedy"]},
    "ucb_vs_ucb": {"agents": ["ucb_u", "ucb_u"]},
    "uts_vs_uts": {"agents": ["uts", "uts"]},
    "eps_vs_eps_vs_eps": {"agents": ["epsilon_greedy", "epsilon_greedy", "epsilon_greedy"]},
    "ucb_vs_ucb_vs_ucb": {"agents": ["ucb_u", "ucb_u", "ucb_u"]},
    "uts_vs_uts_vs_uts": {"agents": ["uts", "uts", "uts"]}
}


stds = ["0", "0.001", "0.005", "0.01", "0.02", "0.05", "0.1"]
corrs = ["0.0"]


def load_case(name):
    all_results = {}
    for std in stds:
        for corr in corrs:
            print("--------------------------------------------------")
            print(f"Loading results for std={std}, corr={corr}")
            filepath = (
                f"../competition_results/{name}/std_{std}_corr_{corr}/results.pkl"
            )
            try:
                results = load_results(filepath)
                all_results[(std, corr)] = results
                print(f"Loaded results for std={std}, corr={corr}")
            except FileNotFoundError:
                raise ValueError(f"Results for std={std}, corr={corr} not found.")
    return all_results


def produce_sample_plot(all_results, std="0.02", corr="0.0", algorithm_name="eps_vs_eps"):
    result = all_results[(std, corr)]
    all_prices = result[0]
    all_rewards = result[1]
    nash_price = result[2]
    nash_reward = result[3]
    no_sims, iterations, n_agents = all_prices.shape

    avg_prices = np.mean(all_prices, axis=0)  # Shape: (iterations, n_agents)
    avg_rewards = np.mean(all_rewards, axis=0)  # Shape: (iterations, n_agents)

    nash_margin = nash_price - PARAMS["c"]
    avg_margins = (avg_prices - PARAMS["c"])/nash_margin  # Shape: (iterations, n_agents)
    avg_rewards = avg_rewards/nash_reward  # Shape: (iterations, n_agents)

    plot_dir = "../sample_competition_plots"
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.rcParams.update({'font.size': 19})
    plt.rcParams.update({'legend.fontsize': 17})

    # Plot the average margins
    for agent_idx in range(n_agents):
        plt.plot(avg_margins[:, agent_idx], label=f"Agent {agent_idx+1} Margin")
    plt.axhline(y=1, color='r', linestyle='--', label='Nash Margin')
    plt.ylim([0.93, 1.17])
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Margin")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{plot_dir}/{algorithm_name}_std_{std}_corr_{corr}_margins.png")
    plt.close()


if __name__ == "__main__":
    for case_name in cases_dict.keys():
        print(f"Processing case: {case_name}")
        all_results = load_case(case_name)
        produce_sample_plot(all_results, algorithm_name=case_name)