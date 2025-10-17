"""Analyse the competition results and generate plots.

Specifically, it analyses the competitions for a signle agent 
and check the reward and margin vs the std of noise."""

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
    "eps":{"agents": ["epsilon_greedy"]},
    "ucb":{"agents": ["ucb_u"]},
    "uts":{"agents": ["uts"]}
}


stds = ["0", "0.001", "0.005", "0.01", "0.02", "0.05", "0.1"]
corrs = ["0.0"]

naming_dict = {
    "eps": "SW-$\\varepsilon$-U",
    "ucb": "SW-UCB-U",
    "uts": "SW-UTS"
}


def load_case(name):
    print("==================================================")
    print(f"Analysing case '{name}'")
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


def compute_avergage_margins(result, c):
    all_prices = result[0]
    all_rewards = result[1]
    nash_price = result[2]
    nash_reward = result[3]
    no_sims, iterations, n_agents = all_prices.shape

    nash_margin = nash_price - c

    # Compute the average margin for each simulation
    margins = (all_prices - c)   # Shape: (no_sims, iterations, n_agents)
    
    avg_margins_per_episode = margins.mean(axis=1)
    avg_margins = avg_margins_per_episode.mean(axis=0)/nash_margin
    se_avg_margins = avg_margins_per_episode.std(axis=0)/np.sqrt(no_sims)/nash_margin
    sd_margins = margins.std(axis=(0, 1))/nash_margin

    avg_rewards_per_episode = all_rewards.mean(axis=1)
    avg_rewards = avg_rewards_per_episode.mean(axis=0)/nash_reward
    se_avg_rewards = avg_rewards_per_episode.std(axis=0)/np.sqrt(no_sims)/nash_reward
    sd_rewards = all_rewards.std(axis=(0,1))/nash_reward


    return {
        "avg_margins": avg_margins,
        "se_avg_margins": se_avg_margins,
        "sd_margins": sd_margins,
        "avg_rewards": avg_rewards,
        "se_avg_rewards": se_avg_rewards,
        "sd_rewards": sd_rewards,
        "nash_margin": nash_margin,
        "nash_reward": nash_reward,
    }

def analyse_case(name):
    print("==================================================")
    print(f"Analysing case '{name}'")
    # Load data
    if name not in cases_dict:
        raise ValueError(f"Case '{name}' not found in cases definition.")
    all_results = load_case(name)

    output_dir = f"../competition_analysis_1/{name}/"
    os.makedirs(output_dir, exist_ok=True)

    # Find the NASH price for the given parameters
    nash_price, nash_reward = find_nash(
        N=1, a=PARAMS["a"], c=PARAMS["c"], mu=PARAMS["mu"], a_0=PARAMS["a_0"]
    )
    c = PARAMS["c"]

    print("Nash margin:", nash_price - c)
    print("Discretisation step:", 1/100/(nash_price - c))

    # Analyse results
    margin_results = {}
    for (std, corr), result in all_results.items():
        margin_results[(std, corr)] = compute_avergage_margins(result, c)

    

    # Collate the results fro corr=0.0
    std_values = []
    avg_margin_values = []
    sd_margin_values = []
    se_avg_margin_values = []
    avg_reward_values = []
    se_avg_reward_values = []
    sd_reward_values = []

    for std in stds:
        std_values.append(float(std))
        avg_margin_values.append(margin_results[(std, "0.0")]["avg_margins"][0])
        sd_margin_values.append(margin_results[(std, "0.0")]["sd_margins"][0])
        se_avg_margin_values.append(margin_results[(std, "0.0")]["se_avg_margins"][0])
        avg_reward_values.append(margin_results[(std, "0.0")]["avg_rewards"][0])
        sd_reward_values.append(margin_results[(std, "0.0")]["sd_rewards"][0])
        se_avg_reward_values.append(margin_results[(std, "0.0")]["se_avg_rewards"][0])


    return {
        "stds": np.array(std_values),
        "avg_margins": np.array(avg_margin_values),
        "sd_margins": np.array(sd_margin_values),
        "se_avg_margins": np.array(se_avg_margin_values),
        "avg_rewards": np.array(avg_reward_values),
        "sd_rewards": np.array(sd_reward_values),
        "se_avg_rewards": np.array(se_avg_reward_values),
        "nash_margin": nash_price - c,
        "nash_reward": nash_reward,
    }


def plot_results_by_case(results_by_case):
    "Plot average margins and rewards for all cases"

    # Increase the font size for better readability
    plt.rcParams.update({'font.size': 19})
    plt.figure()
    for case_name, results in results_by_case.items():
        plt.plot(
            results["stds"],
            results["avg_margins"],
            marker="o",
            label=naming_dict[case_name],
        )
        plt.fill_between(
            results["stds"],
            results["avg_margins"] - results["se_avg_margins"],
            results["avg_margins"] + results["se_avg_margins"],
            alpha=0.2
        )
    plt.xlabel("$\\sigma$")
    plt.ylabel("Average Margin")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../competition_analysis_1/average_margins_all_cases.png")
    # plt.show()
    plt.close()

    # Plot average rewards for all cases
    plt.figure()
    for case_name, results in results_by_case.items():
        plt.plot(
            results["stds"],
            results["avg_rewards"],
            marker="o",
            label=naming_dict[case_name],
        )
        plt.fill_between(
            results["stds"],
            results["avg_rewards"] - results["se_avg_rewards"],
            results["avg_rewards"] + results["se_avg_rewards"],
            alpha=0.2
        )
    plt.xlabel("$\\sigma$")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../competition_analysis_1/average_rewards_all_cases.png")
    # plt.show()
    plt.close()

    # Plot std of margins for all cases
    plt.figure()
    for case_name, results in results_by_case.items():
        plt.plot(
            results["stds"],
            results["sd_margins"],
            marker="o",
            label=naming_dict[case_name],
        )
    plt.xlabel("$\\sigma$")
    plt.ylabel("Std of Margin")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../competition_analysis_1/std_of_margins_all_cases.png")
    # plt.show()
    plt.close()


    # Plot all together on a single suplot with three columns
    plt.rcParams.update({'font.size': 19})
    plt.rcParams.update({'legend.fontsize': 17})
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    for case_name, results in results_by_case.items():
        axs[0].plot(
            results["stds"],
            results["avg_margins"],
            marker="o",
            label=naming_dict[case_name],
        )
        axs[0].fill_between(
            results["stds"],
            results["avg_margins"] - results["se_avg_margins"],
            results["avg_margins"] + results["se_avg_margins"],
            alpha=0.2
        )
        axs[1].plot(
            results["stds"],
            results["sd_margins"],
            marker="o",
            label=naming_dict[case_name],
        )
        axs[2].plot(
            results["stds"],
            results["avg_rewards"],
            marker="o",
            label=naming_dict[case_name],
        )
        axs[2].fill_between(
            results["stds"],
            results["avg_rewards"] - results["se_avg_rewards"],
            results["avg_rewards"] + results["se_avg_rewards"],
            alpha=0.2
        )
    axs[0].set_xlabel("$\\sigma$")
    axs[0].set_ylabel("Average margin")
    axs[0].legend()
    axs[0].grid()
    axs[0].set_ylim(0.988, 1.012)
    axs[1].set_xlabel("$\\sigma$")
    axs[1].set_ylabel("SD of margin")
    axs[1].legend()
    axs[1].grid()   
    axs[2].set_xlabel("$\\sigma$")
    axs[2].set_ylabel("Average reward")
    axs[2].legend()
    axs[2].grid()
    plt.tight_layout()
    plt.savefig("../competition_analysis_1/summary_all_cases_comp_1_stat.png")
    plt.close()
        


if __name__ == "__main__":
    results_by_case = {}
    for case_name in cases_dict.keys():
        results_by_case[case_name] = analyse_case(case_name)

    plot_results_by_case(results_by_case)