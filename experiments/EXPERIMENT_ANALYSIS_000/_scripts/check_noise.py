"""Test the noisiness of the evaluation results."""

import os
from tools.evaluation.evaluate import evaluate_parameters
from tools.bandits.sw_eps import SW_EpsilonGreedy
from tools.bandits.sw_ucb_u import SW_UCB_U
from tools.bandits.sw_uts import SW_UTS


import pandas as pd
import numpy as np


EPS_GREEDY_PARAMS = {
    "sliding_window": 30,
    "epsilon": 0.3,
    "no_actions": 100,
    "no_neighbors": 7,
}

UCB_U_PARAMS = {
    "sliding_window": 30,
    "no_actions": 100,
    "no_neighbors": 3,
    "alpha": 0.1,
}

UTS_PARAMS = {
    "sliding_window": 30,
    "variance": 0.02,
    "no_actions": 100,
    "no_neighbors": 3,
}

if __name__ == "__main__":
    # Run the evaluation for a set of parameters for 100 times to see the variance in the results
    agent_type = SW_EpsilonGreedy
    parameters = EPS_GREEDY_PARAMS

    agent_type = SW_UCB_U
    parameters = UCB_U_PARAMS

    agent_type = SW_UTS
    parameters = UTS_PARAMS

    noise_level = 0.1
    no_iterations = 500

    results = []
    for i in range(100):
        avg_reward = evaluate_parameters(
            agent_type, parameters, no_iterations=no_iterations, noise_level=noise_level
        )
        results.append(avg_reward["average_regret"])
        print(f"Run {i+1}/100: Average Reward = {avg_reward["average_regret"]}")
    results = np.array(results)
    print(f"Mean Average Reward over 100 runs: {np.mean(results)}")
    print(f"Standard Deviation of Average Reward over 100 runs: {np.std(results)}")
    print(f"Results: {results}")
