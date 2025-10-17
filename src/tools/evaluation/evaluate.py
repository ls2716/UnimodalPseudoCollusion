"""Implement testing suite for the algorithms in the ExplorationCollusion project."""

import numpy as np
from tools.evaluation.evaluate_nonstationary_triangle import evaluate_algorithm as evaluate_nonstationary
from tools.evaluation.evaluate_stationary_triangle import evaluate_algorithm as evaluate_stationary
import pandas as pd
import tqdm

def evaluate_agent(agent, noise_level, no_iterations, no_sims):
    """Evaluate the agent for both stationary and non-stationary environments."""

    # Evaluate in stationary environment
    regrets = []
    for sim in range(no_sims):
        regret = evaluate_stationary(agent, no_iterations, noise_level)
        regrets.append(regret)
    # Average regret over simulations
    stationary_regret = np.mean(regrets)

    # Evaluate in non-stationary environment
    regrets = []
    for sim in range(no_sims):
        regret = evaluate_nonstationary(agent, no_iterations, noise_level)
        regrets.append(regret)
    # Average regret over simulations
    nonstationary_regret = np.mean(regrets)

    results = {
        "stationary_regret": stationary_regret,
        "nonstationary_regret": nonstationary_regret,
        "average_regret": (stationary_regret + nonstationary_regret) / 2,
    }

    return results

def evaluate_parameters(agent_type, parameters, noise_level, no_iterations=500, no_sims=50):
    """Evaluate the agent with given parameters."""
    agent = agent_type(
        **parameters,  # Unpack parameters into the agent constructor
        starting_action=0.
    )
    result = evaluate_agent(agent, noise_level, no_iterations, no_sims)
    return result


def evaluate_parameter_samples(parameter_df, agent_type, noise_level, no_iterations=500, no_sims=50):
    """Evaluate a dataframe of parameters."""
    results = []
    for index, row in tqdm.tqdm(parameter_df.iterrows(), total=parameter_df.shape[0], desc="Evaluating parameters"):
        parameters = row.to_dict()
        result = evaluate_parameters(agent_type, parameters, noise_level, no_iterations, no_sims)
        results.append((result["stationary_regret"], result["nonstationary_regret"], result["average_regret"]))
    results_df = pd.DataFrame(results, columns=["stationary_regret", "nonstationary_regret", "average_regret"])
    results_df.index = parameter_df.index
    return results_df




# def main():
#     from sw_eps import SW_EpsilonGreedy

#     # Define parameters
#     noise_level = 0.05
#     no_iterations = 500
#     period = 1000
#     no_actions = 21
#     no_neighbors = 1
#     sliding_window = 10
#     epsilon = 1.0

#     # Initialize agent
#     agent = SW_EpsilonGreedy(
#         epsilon=epsilon,
#         no_actions=no_actions,
#         no_neighbors=no_neighbors,
#         sliding_window=sliding_window,
#         starting_action=0,
#     )

#     # Evaluate the agent
#     results = evaluate_agent(agent, noise_level, no_iterations)

#     print(f"Results for SW_EpsilonGreedy with noise level {noise_level}:")
#     print(f"Stationary Regret: {results['stationary_regret']:.4f}")
#     print(f"Non-Stationary Regret: {results['nonstationary_regret']:.4f}")
#     print(f"Average Regret: {results['average_regret']:.4f}")


# if __name__ == "__main__":
#     import cProfile
#     import pstats
#     import io

#     pr = cProfile.Profile()
#     pr.enable()
#     main()
#     pr.disable()
#     s = io.StringIO()
#     sortby = "cumulative"
#     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#     ps.print_stats(20)  # print top 20 lines
#     print("\n--- Profiling Results (Top 20 by cumulative time) ---")
#     print(s.getvalue())
