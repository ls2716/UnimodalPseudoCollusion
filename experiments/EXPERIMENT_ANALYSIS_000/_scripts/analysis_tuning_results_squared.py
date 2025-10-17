import numpy as np
import os
import pandas as pd


def read_data(algorithm, noise_level, testcase):
    """Read evaluation results from a CSV file."""
    filepath = os.path.join(
        "..",
        "tuning",
        f"{algorithm}",
        f"noise_{noise_level}",
        "evaluation_results.csv",
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    df = pd.read_csv(filepath, index_col=0)
    if testcase not in df.columns:
        raise ValueError(f"Testcase '{testcase}' not found in results.")
    return df[testcase]


if __name__ == "__main__":
    noises = [0, 0.02, 0.05, 0.07, 0.1]
    algorithms = ["epsilon_greedy", "ucb_u", "uts"]
    testcase = "average_regret"

    # Create a disctionary to hold the data
    data = {
        alg: {noise: read_data(alg, noise, testcase) for noise in noises}
        for alg in algorithms
    }
    dataframes = {alg: pd.DataFrame(data[alg]) for alg in algorithms}

    # Normalise the dataframes by the min value across samples and algorithms
    for noise in noises:
        min_value = min(dataframes[alg][noise].min() for alg in algorithms)
        for alg in algorithms:
            dataframes[alg][f"normalized_noise_{noise}"] = (
                dataframes[alg][noise] - min_value
            )

    # Compute the squared averages for each algorithm
    for alg in algorithms:
        sums = np.zeros(len(dataframes[alg]))
        for noise in noises:
            sums += dataframes[alg][f"normalized_noise_{noise}"] ** 2
        dataframes[alg]["combined_squared"] = np.sqrt(sums / len(noises))

    # Drop the normalized columns
    for alg in algorithms:
        dataframes[alg] = dataframes[alg].drop(
            columns=[f"normalized_noise_{noise}" for noise in noises]
        )

        # Save the dataframes to CSV files
    output_folder = os.path.join("../tuning_results_analysis")
    os.makedirs(output_folder, exist_ok=True)
    for alg, df in dataframes.items():
        output_filepath = os.path.join(output_folder, f"{alg}_analysis.csv")
        df.to_csv(output_filepath, index_label="index")
        print(f"Saved analysis results for {alg} to {output_filepath}")

    # Compute the 1% percentile for each algorithm and each column
    quantiles = {
        alg: {col: df[col].quantile(0.01) for col in df.columns}
        for alg, df in dataframes.items()
    }

    # Create a latex table with the top 1% percentile for each algorithm and each column
    # Noise in rows as well as the combined squared column
    # Algorithms in columns with the top 1% percentile values
    latex_table = "\\begin{tabular}{lccc}\n"
    latex_table += "Noise Std $\sigma$ & SW-\\varepsilon-U & SW-UCB-U & SW-UTS \\\\\n"
    latex_table += "\\hline\n"
    for noise in noises:
        latex_table += f"{noise} & {quantiles['epsilon_greedy'][noise]:.4f} & {quantiles['ucb_u'][noise]:.4f} & {quantiles['uts'][noise]:.4f} \\\\\n"
    latex_table += f"Combined Squared & {quantiles['epsilon_greedy']['combined_squared']:.4f} & {quantiles['ucb_u']['combined_squared']:.4f} & {quantiles['uts']['combined_squared']:.4f} \\\\\n"
    latex_table += "\\end{tabular}\n"

    print("Latex table of 1% percentiles:")
    print(latex_table)

    # Print the best regret values for each algorithm and each noise level
    best_regrets = {
        alg: {col: df[col].min() for col in df.columns} for alg, df in dataframes.items()
    }


    # Print a Latex table with the best regret values for each algorithm and each noise level
    latex_table_best = "\\begin{tabular}{lccc}\n"
    latex_table_best += "Noise Std $\sigma$ & SW-$\\varepsilon$-U & SW-UCB-U & SW-UTS \\\\\n"
    latex_table_best += "\\hline\n"
    for noise in noises:
        latex_table_best += f"{noise} & {best_regrets['epsilon_greedy'][noise]:.4f} & {best_regrets['ucb_u'][noise]:.4f} & {best_regrets['uts'][noise]:.4f} \\\\\n"
    latex_table_best += f"$r^o$ & {best_regrets['epsilon_greedy']['combined_squared']:.4f} & {best_regrets['ucb_u']['combined_squared']:.4f} & {best_regrets['uts']['combined_squared']:.4f} \\\\\n"
    latex_table_best += "\\end{tabular}\n" 

    print("Latex table of best regret values:")
    print(latex_table_best)


    # Comput the average regret for each algorithm and each noise level