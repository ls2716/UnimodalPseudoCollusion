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

    # Compute the average regret for each algorithm across noise levels
    for alg in algorithms:
        sums = np.zeros(len(dataframes[alg]))
        for noise in noises:
            sums += dataframes[alg][noise] 
        dataframes[alg]["combined_average"] = sums / len(noises)
    


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
    latex_table_best += f"$r^o$ & {best_regrets['epsilon_greedy']['combined_average']:.4f} & {best_regrets['ucb_u']['combined_average']:.4f} & {best_regrets['uts']['combined_average']:.4f} \\\\\n"
    latex_table_best += "\\end{tabular}\n" 

    print("Latex table of best regret values:")
    print(latex_table_best)


