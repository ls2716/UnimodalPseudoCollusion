import numpy as np
import pandas as pd
import os

from tools.evaluation.evaluate import evaluate_parameters
from tools.bandits.sw_eps import SW_EpsilonGreedy
from tools.bandits.sw_ucb_u import SW_UCB_U
from tools.bandits.sw_uts import SW_UTS

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
    "std": 0.02,
    "no_actions": 100,
    "no_neighbors": 3,
}


agent_type_dict = {
    "epsilon_greedy": {
        "agent_type": SW_EpsilonGreedy,
        "default_params": EPS_GREEDY_PARAMS,
    },
    "ucb_u": {"agent_type": SW_UCB_U, "default_params": UCB_U_PARAMS},
    "uts": {"agent_type": SW_UTS, "default_params": UTS_PARAMS},
}


def evaluate(agent_type, parameters, no_iterations=500, noise_level=0.1):
    """Evaluate the agent with given parameters."""
    result = evaluate_parameters(
        agent_type,
        parameters,
        no_iterations=no_iterations,
        noise_level=noise_level,
        no_sims=1,
    )
    return result


def evaluate_until_convergent(
    agent_type,
    parameters,
    noise_level,
    no_iterations,
    min_accuracy=0.02,
    max_evaluations=1000,
    start_evaluations=100):
    """Evaluate the agent until it reaches the target regret or max iterations."""
    stationary_regrets = []
    nonstationary_regrets = []
    avg_regrets = []
    for it in range(max_evaluations):
        result = evaluate(
            agent_type,
            parameters,
            no_iterations=no_iterations,
            noise_level=noise_level,
        )
        stationary_regrets.append(result["stationary_regret"])
        nonstationary_regrets.append(result["nonstationary_regret"])
        avg_regrets.append(result["average_regret"])

        if it>0 and it%100==0:
            sqrt_it = np.sqrt(it)
            stationary_okay = np.std(stationary_regrets)/np.mean(stationary_regrets)/sqrt_it < min_accuracy
            nonstationary_okay = np.std(nonstationary_regrets)/np.mean(nonstationary_regrets)/sqrt_it < min_accuracy
            avg_okay = np.std(avg_regrets)/np.mean(avg_regrets)/sqrt_it < min_accuracy
            print(f"Eval {it}: stationary {np.mean(stationary_regrets):.4f}±{np.std(stationary_regrets)/sqrt_it:.4f} {'OK' if stationary_okay else 'NO'};\n"
                  f"nonstationary {np.mean(nonstationary_regrets):.4f}±{np.std(nonstationary_regrets)/sqrt_it:.4f} {'OK' if nonstationary_okay else 'NO'};\n"
                  f"avg {np.mean(avg_regrets):.4f}±{np.std(avg_regrets)/sqrt_it:.4f} {'OK' if avg_okay else 'NO'}")
            if stationary_okay and nonstationary_okay and avg_okay and it>start_evaluations:
                print(f"Converged after {it} evaluations.")
                break
    return {
        "stationary_regret": np.mean(stationary_regrets),
        "stationary_std": np.std(stationary_regrets)/sqrt_it,
        "nonstationary_regret": np.mean(nonstationary_regrets),
        "nonstationary_std": np.std(nonstationary_regrets)/sqrt_it,
        "average_regret": np.mean(avg_regrets),
        "average_std": np.std(avg_regrets)/sqrt_it,
        "evaluations": it+1,}


if __name__ == "__main__":
    results_folder = "../algorithm_evaluations"
    os.makedirs(results_folder, exist_ok=True)
    result_filepath_template = os.path.join(results_folder, "results_{agent_type}.csv")


    T_values = [100, 200, 300, 400, 500, 1000]
    noise_values = [0.0, 0.05, 0.1]

    agent_name = "epsilon_greedy"  # Change to "ucb_u" for UCB-U agent or "uts" for UTS agent
    agent_info = agent_type_dict[agent_name]
    agent_type = agent_info["agent_type"]
    default_params = agent_info["default_params"]


    for agent in ["uts"]:
        agent_info = agent_type_dict[agent]
        agent_type = agent_info["agent_type"]
        default_params = agent_info["default_params"]
        results_filepath = result_filepath_template.format(agent_type=agent)
        
        all_results = []
        for T in T_values:
            for noise_level in noise_values:
                print(f"Evaluating {agent} with T={T}, noise_level={noise_level}")
                result = evaluate_until_convergent(
                    agent_type,
                    default_params,
                    noise_level=noise_level,
                    no_iterations=T,)
                result_row = {
                    "T": T,
                    "noise_level": noise_level,
                    **result
                }
                all_results.append(result_row)
        

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_filepath, index=False)
        print(f"Results saved to {results_filepath}")

    result = evaluate_until_convergent(
        agent_type,
        default_params,
        noise_level=0.,
        no_iterations=500,)
    
    # print(f"Final result for {agent_name} with default parameters: {result}")
