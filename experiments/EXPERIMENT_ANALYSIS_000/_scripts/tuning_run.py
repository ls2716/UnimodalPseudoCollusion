from tools.evaluation.evaluate import evaluate_parameter_samples
import pandas as pd
from tools.bandits.sw_eps import SW_EpsilonGreedy
from tools.bandits.sw_ucb_u import SW_UCB_U
from tools.bandits.sw_uts import SW_UTS

import os


def load_parameters(samples_filepath):
    """Load parameters from a CSV file based on the experiment name."""
    if not os.path.exists(samples_filepath):
        raise FileNotFoundError(f"Parameter file not found: {samples_filepath}")
    return pd.read_csv(samples_filepath, index_col=0)


agent_name_dict = {
    "epsilon_greedy": {
        "agent_type": SW_EpsilonGreedy,
    },
    "ucb_u": {
        "agent_type": SW_UCB_U,
    },
    "uts": {
        "agent_type": SW_UTS,
    },
}

samples_filepath_template = "../samples/{agent_name}_samples.csv"
results_filepath_template = "../tuning/{agent_name}/noise_{noise_level}"


if __name__ == "__main__":
    # HARDCODED PARAMETERS
    agent_name = "uts"  # Change to "ucb_u" for UCB-U agent or "uts" for UTS agent
    agent_type = agent_name_dict[agent_name]["agent_type"]
    samples_filepath = samples_filepath_template.format(agent_name=agent_name)

    # Set this to True to always recompute all results from scratch
    recreate_result = False

    # Load parameters from the specified experiment
    parameter_df = load_parameters(samples_filepath)

    print("Loaded parameters:")
    print(parameter_df.head())  # Display the first few rows of the parameters dataframe

    noise_levels = [0.1, 0, 0.05, 0.02, 0.07]  # Define the noise levels for evaluation

    for noise_level in noise_levels:
        print(f"Evaluating with noise level: {noise_level}")
        # Create a folder to hold the results
        results_folder = results_filepath_template.format(
            agent_name=agent_name, noise_level=noise_level
        )
        os.makedirs(results_folder, exist_ok=True)
        print(f"Results will be saved in: {results_folder}")

        # Try to load the results if they already exist
        old_results_path = os.path.join(results_folder, "evaluation_results.csv")
        if os.path.exists(old_results_path):
            old_results_df = pd.read_csv(old_results_path, index_col=0)
            print("Loaded existing results:")
        else:
            old_results_df = None
            print("No existing results found, starting fresh.")

        full_results = None
        if old_results_df is not None:
            full_results = old_results_df
        else:
            # If no previous results, start with an empty DataFrame
            full_results = pd.DataFrame(
                columns=["stationary_regret", "nonstationary_regret", "average_regret"]
            )

        no_total_samples = parameter_df.shape[0]
        # no_total_samples = 1024
        print(f"Total number of samples to evaluate: {no_total_samples}")
        print("Evaluating parameters in batches of 64...")
        batch_size = 64

        # For each batch, only evaluate parameter indices not already in results (if not recreating)
        for start in range(0, no_total_samples, batch_size):
            end = min(start + batch_size, no_total_samples)
            batch_indices = parameter_df.index[start:end]
            if not recreate_result:
                # Only evaluate indices not already in results
                indices_to_eval = [
                    idx for idx in batch_indices if idx not in full_results.index
                ]
                if not indices_to_eval:
                    continue  # skip this batch
                batch_df = parameter_df.loc[indices_to_eval]
            else:
                batch_df = parameter_df[start:end]
            if batch_df.empty:
                print(f"Batch {start // batch_size + 1} is empty, skipping.")
                continue
            results = evaluate_parameter_samples(
                batch_df, agent_type, noise_level, no_sims=50
            )
            # Save the results to csv file
            full_results = pd.concat([full_results, results])
            full_results = full_results[~full_results.index.duplicated(keep="last")]
            # Sort the results by index to maintain order
            full_results.sort_index(inplace=True)
            # Save results to a CSV file
            full_results.to_csv(os.path.join(results_folder, "evaluation_results.csv"))

            # Print the top few results for verification
            # Sort by average regret and print the top 5 results
            print("Top 5 results so far:")
            best_results = full_results.sort_values(by="average_regret").head(5)
            # Get the corresponding parameters
            best_parameters = parameter_df.loc[best_results.index]
            # Print the best results with parameters
            df_to_print = pd.concat([best_results, best_parameters], axis=1)
            print(df_to_print)
