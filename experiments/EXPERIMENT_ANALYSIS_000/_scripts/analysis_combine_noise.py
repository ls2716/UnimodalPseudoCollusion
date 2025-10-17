"""Create an csv with evaluation that creates a weighted average of regrets for different noises."""
import os
import pandas as pd
import numpy as np


def load_evaluation_results(results_filepath):
    if not os.path.exists(results_filepath):
        raise FileNotFoundError(f"Results file not found: {results_filepath}")
    return pd.read_csv(results_filepath, index_col=0)


def load_samples(samples_filepath):
    if not os.path.exists(samples_filepath):
        raise FileNotFoundError(f"Parameter file not found: {samples_filepath}")
    return pd.read_csv(samples_filepath, index_col=0)


def produce_combined_regret_dataframe(agent_name, case, method, no_samples=None, noise_levels=None, weights=None, base_dir=None):
    """
    Produces a DataFrame with the weighted average regret for a given agent, case, and method.
    
    Parameters:
        agent_name (str): "epsilon_greedy", "ucb_u", or "uts"
        case (str): "stationary", "nonstationary", or "average"
        method (str): "squared", "arithmetic", or "max"
        noise_levels (list, optional): List of noise levels. Defaults to [0, 0.02, 0.05, 0.07, 0.1].
        weights (list, optional): List of weights for each noise level. Defaults to equal weights.
        base_dir (str, optional): Base directory for file paths. Defaults to current script location.
        
    Returns:
        pd.DataFrame: DataFrame with parameter sets and their weighted average regret.
    """
    agent_name_dict = {
        "epsilon_greedy": {"agent_type": "SW_EpsilonGreedy"},
        "ucb_u": {"agent_type": "SW_UCB_U"},
        "uts": {"agent_type": "SW_UTS"},
    }
    if agent_name not in agent_name_dict:
        raise ValueError(f"Unknown agent_name: {agent_name}")

    if noise_levels is None:
        noise_levels = [0, 0.02, 0.05, 0.07, 0.1]
    if weights is None:
        weights = [1.0 for _ in noise_levels]
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    samples_filepath_template = os.path.join(base_dir, "../samples/{agent_name}_samples.csv")
    results_filepath_template = os.path.join(base_dir, "../tuning_contabo/{agent_name}/noise_{noise_level}/evaluation_results.csv")

    methods = {
        "squared": lambda x, w: np.sqrt((x**2 * w).sum() / sum(w)),
        "arithmetic": lambda x, w: (x * w).sum() / sum(w),
        "max": lambda x, w: x.max(),
    }
    if case not in ["stationary", "nonstationary", "average"]:
        raise ValueError(f"Unknown case: {case}")
    
    
    # Support for method='noise_{noise_level}'
    if method.startswith("noise_"):
        noise_level = method.split("_", 1)[1]
        combined_results = load_evaluation_results(
            results_filepath_template.format(agent_name=agent_name, noise_level=noise_level)
        )
        # Drop all columns except the relevant regret column
        combined_results = combined_results[[f"{case}_regret"]]
        combined_results.rename(columns={f"{case}_regret": "average_regret"}, inplace=True)
    else:
        if method not in methods:
            raise ValueError(f"Unknown method: {method}")
    
        # Load evaluation results for all noise levels
        combined_results = pd.DataFrame()
        for noise_level in noise_levels:
            results_path = results_filepath_template.format(agent_name=agent_name, noise_level=noise_level)
            if not os.path.exists(results_path):
                raise FileNotFoundError(f"Results file not found: {results_path}")
            df = pd.read_csv(results_path)
            combined_results[f"avg_regret_{noise_level}"] = df[f"{case}_regret"]

        # Normalise the regrets relative to the best regret for each noise level
        for noise_level in noise_levels:
            col = f"avg_regret_{noise_level}"
            min_regret = combined_results[col].min()
            combined_results[col] = combined_results[col] - min_regret

        # Compute the weighted average regret using the specified method
        combined_results["average_regret"] = combined_results.apply(
            lambda row: methods[method](
                row[[f"avg_regret_{nl}" for nl in noise_levels]].values, weights
            ),
            axis=1,
        )

    # Load samples for the agent
    samples_filepath = samples_filepath_template.format(agent_name=agent_name)
    if not os.path.exists(samples_filepath):
        raise FileNotFoundError(f"Parameter file not found: {samples_filepath}")
    parameter_df = pd.read_csv(samples_filepath, index_col=0)

    parameter_df = parameter_df.copy()
    if no_samples is not None:
        parameter_df = parameter_df.iloc[:no_samples]
        combined_results = combined_results.iloc[:no_samples]
    parameter_df["average_regret"] = combined_results["average_regret"].values

    return parameter_df


if __name__ == "__main__":

    # Example usage
    agent_name = "epsilon_greedy"  # Change to "epsilon_greedy" for "epsilon_greedy", "ucb_u", or "uts"
    case = "average"  # Change to "stationary", "nonstationary", or "average"
    method = "squared"  # Change to "squared", "arithmetic", or "max"

    df = produce_combined_regret_dataframe(agent_name, case, method)

    print(df.head())
    print(df.describe())