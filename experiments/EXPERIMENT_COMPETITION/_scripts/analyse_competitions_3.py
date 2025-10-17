"""Analyse the competition results and generate plots.

Specifically, it analyses the competitions between three agents under different noise levels
expressed as standard deviation."""

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
    "eps_vs_eps_vs_eps": {"agents": ["epsilon_greedy", "epsilon_greedy", "epsilon_greedy"]},
    "ucb_vs_ucb_vs_ucb": {"agents": ["ucb_u", "ucb_u", "ucb_u"]},
    "uts_vs_uts_vs_uts": {"agents": ["uts", "uts", "uts"]},
}


stds = ["0", "0.001", "0.005", "0.01", "0.02", "0.05", "0.1"]
corrs = ["0.0"]

naming_dict = {
    "eps_vs_eps_vs_eps": "3 SW-$\\varepsilon$-U",
    "ucb_vs_ucb_vs_ucb": "3 SW-UCB-U",
    "uts_vs_uts_vs_uts": "3 SW-UTS",
}


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


def compute_average_margins(result, c):
    all_prices = result[0]
    all_rewards = result[1]
    nash_price = result[2]
    nash_reward = result[3]
    no_sims, iterations, n_agents = all_prices.shape

    nash_margin = nash_price - c

    margins = (all_prices - c)   # Shape: (no_sims, iterations, n_agents)
    # Compute average margin per episode and SE of the average margin
    avg_margins_per_episode = margins.mean(axis=1)
    avg_margins = avg_margins_per_episode.mean(axis=0)/nash_margin
    se_avg_margins = avg_margins_per_episode.std(axis=0)/np.sqrt(no_sims)/nash_margin
    # Compute SD of margins
    sd_margins = margins.std(axis=(0, 1))/nash_margin
    # Compute correlation of margins between agents
    corr_margins = np.corrcoef(
        margins.reshape(-1, n_agents), rowvar=False
    )
    avg_rewards_per_episode = all_rewards.mean(axis=1)
    avg_rewards = avg_rewards_per_episode.mean(axis=0)/nash_reward
    se_avg_rewards = avg_rewards_per_episode.std(axis=0)/(np.sqrt(no_sims)*nash_reward)

    return {
        "avg_margins": avg_margins,
        "se_avg_margins": se_avg_margins,
        "sd_margins": sd_margins,
        "corr_margins": corr_margins[0, 1],
        "avg_rewards": avg_rewards,
        "se_avg_rewards": se_avg_rewards,
    }

def analyse_case(name):
    print("==================================================")
    print(f"Analysing case '{name}'")
    # Load data
    if name not in cases_dict:
        raise ValueError(f"Case '{name}' not found in cases definition.")
    all_results = load_case(name)
    print(f"Loaded all results for case '{name}'")

    output_dir = f"../competition_analysis_3/{name}/"
    os.makedirs(output_dir, exist_ok=True)

    # Find the NASH price for the given parameters
    nash_price, nash_reward = find_nash(
        N=2, a=PARAMS["a"], c=PARAMS["c"], mu=PARAMS["mu"], a_0=PARAMS["a_0"]
    )
    c = PARAMS["c"]

    # Analyse results
    case_results = {}
    for (std, corr), result in all_results.items():
        case_results[(std, corr)] = compute_average_margins(result, c)


    return case_results



def plot_summary_margins_vs_corr_and_noise(results_by_case):
    # Create a single subplot with average margins for all cases
    # With separate lines for different noise correlations
    plt.rcParams.update({'font.size': 19})
    plt.rcParams.update({'legend.fontsize': 17})
    plt.subplots(1, 3, figsize=(17, 5))

    for i, (case_name, results) in enumerate(results_by_case.items()):
        ax = plt.subplot(1, 3, i+1)
        for corr in corrs:
            std_values = []
            avg_margin_values = []
            se_avg_margin_values = []
            for std in stds:
                if (std, corr) in results:
                    std_values.append(float(std))
                    avg_margin_values.append(
                        results[(std, corr)]["avg_margins"][0]
                    )
                    se_avg_margin_values.append(
                        results[(std, corr)]["se_avg_margins"][0]
                    )
            ax.plot(
                std_values,
                avg_margin_values,
                label=f"$\\rho$={corr}",
                marker="o",
            )
            ax.fill_between(
                std_values,
                np.array(avg_margin_values) - np.array(se_avg_margin_values),
                np.array(avg_margin_values) + np.array(se_avg_margin_values),
                alpha=0.2
            )
        ax.set_title(naming_dict[case_name])
        ax.set_xlabel("$\\sigma$")
        if i == 0:
            ax.set_ylabel("Average Margin")
        ax.legend()
        ax.grid()
    plt.tight_layout()
    plt.savefig("../competition_analysis_3/summary_margins_with_correlations_comp_3.png")
    # plt.show()
    plt.close()


def plot_summary_all_cases(case_results):
    # Create a single subplot with average margins for all cases
    # With separate lines for different noise correlations
    plt.rcParams.update({'font.size': 19})
    plt.rcParams.update({'legend.fontsize': 17})

    # Plot average margins, stds, correlations and rewards on 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    # Plot the results for zero correlation only
    corr = "0.0"
    # Plot the average margins
    for i, (case_name, results) in enumerate(case_results.items()):
        std_values = []
        avg_margin_values = []
        se_avg_margin_values = []
        sd_margin_values = []
        corr_values = []
        avg_reward_values = []
        se_avg_reward_values = []
        for std in stds:
            if (std, corr) in results:
                std_values.append(float(std))
                avg_margin_values.append(
                    results[(std, corr)]["avg_margins"][0]
                )
                se_avg_margin_values.append(
                    results[(std, corr)]["se_avg_margins"][0]
                )
                sd_margin_values.append(
                    results[(std, corr)]["sd_margins"][0]
                )
                corr_values.append(
                    results[(std, corr)]["corr_margins"]
                )
                avg_reward_values.append(
                    results[(std, corr)]["avg_rewards"][0]
                )
                se_avg_reward_values.append(
                    results[(std, corr)]["se_avg_rewards"][0]
                )

        axs[0, 0].plot(
            std_values,
            avg_margin_values,
            label=naming_dict[case_name],
            marker="o",
        )
        axs[0, 0].fill_between(
            std_values,
            np.array(avg_margin_values) - np.array(se_avg_margin_values),
            np.array(avg_margin_values) + np.array(se_avg_margin_values),
            alpha=0.2
        )
        axs[0, 1].plot(
            std_values,
            sd_margin_values,
            label=naming_dict[case_name],
            marker="o",
        )
        axs[1, 0].plot(
            std_values,
            corr_values,
            label=naming_dict[case_name],
            marker="o",
        )
        axs[1, 1].plot(
            std_values,
            avg_reward_values,
            label=naming_dict[case_name],
            marker="o",
        )
        axs[1, 1].fill_between(
            std_values,
            np.array(avg_reward_values) - np.array(se_avg_reward_values),
            np.array(avg_reward_values) + np.array(se_avg_reward_values),
            alpha=0.2
        )
    # No titles but axis names
    axs[0, 0].set_ylabel("Average margin")
    axs[0, 1].set_ylabel("SD of margin")
    axs[1, 0].set_ylabel("Correlation of margins")
    axs[1, 1].set_ylabel("Average reward")
    for ax in axs.flat:
        ax.set_xlabel("$\\sigma$")
        ax.grid()
        ax.legend()
    plt.tight_layout()
    plt.savefig("../competition_analysis_3/summary_all_cases_comp_3.png")


if __name__ == "__main__":
    case_results = {}
    for case_name in cases_dict.keys():
        case_results[case_name] = analyse_case(case_name)

    plot_summary_margins_vs_corr_and_noise(case_results)
    plot_summary_all_cases(case_results)