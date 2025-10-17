"""Create sample plots for the test cases."""

import os
import matplotlib.pyplot as plt

from tools.evaluation.evaluate_stationary_triangle import (
    evaluate_algorithm as evaluate_stationary,
)
from tools.evaluation.evaluate_nonstationary_triangle import (
    evaluate_algorithm as evaluate_nonstationary,
)
from tools.bandits.sw_eps import SW_EpsilonGreedy
from tools.bandits.sw_ucb_u import SW_UCB_U
from tools.bandits.sw_uts import SW_UTS

### HARD CODED
# Definition of agents for the experiments
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
    "std": 0.01,
    "no_actions": 100,
    "no_neighbors": 3,
}


def get_agent(agent_name):
    if agent_name == "epsilon_greedy":
        return SW_EpsilonGreedy(**EPS_GREEDY_PARAMS)
    elif agent_name == "ucb_u":
        return SW_UCB_U(**UCB_U_PARAMS)
    elif agent_name == "uts":
        return SW_UTS(**UTS_PARAMS)
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")


if __name__ == "__main__":
    # Increase the font size for better readability
    # Set axis label size
    plt.rcParams.update({'font.size': 20})

    agent_name = "epsilon_greedy"  # Change to "ucb_u" or "uts" to test other agents

    NO_ITERATIONS = 500
    NOISE_LEVEL = 0.02

    agent = get_agent(agent_name)

    # Evaluate stationary
    stationary_results = evaluate_stationary(
        agent, NO_ITERATIONS, NOISE_LEVEL, gather_data=True
    )
    # Evaluate non-stationary
    agent.reset()  # Reset the agent before re-evaluation
    nonstationary_results = evaluate_nonstationary(
        agent, NO_ITERATIONS, NOISE_LEVEL, gather_data=True
    )

    # Create output directory if it doesn't exist
    output_dir = "../sample_plots/"
    os.makedirs(output_dir, exist_ok=True)

    # Plot stationary  and non-stationary results on subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    # Plot stationary actions and the optimal max action
    axs[0].plot(
        stationary_results["actions"], label="Agent Actions", color='blue', alpha=0.6
    )
    optimal_action = stationary_results["max_action"]
    axs[0].axhline(
        y=optimal_action,
        color="r",
        linestyle="--",
        label=f"Optimal Action {optimal_action:.2f}",
    )

    # Set y-axis limits for better visibility
    axs[0].set_ylim(0, 1)

    # Add labels
    # axs[0].set_title("Stationary Testcase")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Action")
    axs[0].legend()
    axs[0].grid()

    # Plot non-stationary actions and the optimal actions
    axs[1].plot(
        nonstationary_results["actions"], label="Agent Actions", color="blue", alpha=0.6
    )
    # Optimal action changes over time
    optimal_actions = [0.1 + 0.8 * t / NO_ITERATIONS for t in range(NO_ITERATIONS)]
    axs[1].plot(optimal_actions, color="r", linestyle="--", label="Optimal Action")

    # Add labels
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Action")
    # axs[1].set_title("Nonstationary Testcase")
    axs[1].legend()
    axs[1].grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{agent_name}_testcases.png"))
    plt.close()
    print(f"Sample plots saved to {output_dir}")
