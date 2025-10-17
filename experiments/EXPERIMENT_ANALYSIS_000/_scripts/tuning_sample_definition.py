"""Definition of space samples for hyperparameter optimization.

First defines the parameter ranges for each agent, then generates
Sobol samples within these ranges, and finally saves the samples
to CSV files for later use in hyperparameter optimization experiments.

The samples are generated using a Sobol sequence to ensure a good coverage
of the parameter space. The generated samples are saved in a folder
named 'samples' in the parent directory of the current script.

Each agent's parameters are defined in a dictionary, specifying the
minimum and maximum values for each parameter.
"""

import os
import json

import pandas as pd
import matplotlib.pyplot as plt

from tools import sampling

EPS_GREEDY_PARAM_RANGE = {
    "sliding_window": (1, 100),
    "epsilon": (0.01, 1.0),
    "no_actions": (5, 201),
    "no_neighbors": (1, 20),
}

UCB_U_PARAM_RANGE = {
    "sliding_window": (1, 100),
    "alpha": (0.01, 4.0),
    "no_actions": (5, 201),
    "no_neighbors": (1, 20),
}

UTS_PARAM_RANGE = {
    "sliding_window": (1, 100),
    "std": (0.001, 0.2),
    "no_actions": (5, 201),
    "no_neighbors": (1, 20),
}


N_SAMPLES = 2048  # Number of samples to generate
SEED = 2716  # Seed for reproducibility
###

# Create a folder to hold the samples
samples_folder = os.path.join("../samples")
os.makedirs(samples_folder, exist_ok=True)

# Generate Sobol samples for epsilon-greedy parameters
epsilon_greedy_samples, keys = sampling.sobol_prefix_samples(
    EPS_GREEDY_PARAM_RANGE, N_SAMPLES, seed=2716,
)
# Transform a pandas df
epsilon_greedy_samples = pd.DataFrame(
    epsilon_greedy_samples, columns=keys
)
# Correct the data types
epsilon_greedy_samples["sliding_window"] = epsilon_greedy_samples["sliding_window"].astype(int)
epsilon_greedy_samples["no_actions"] = epsilon_greedy_samples["no_actions"].astype(int)
epsilon_greedy_samples["no_neighbors"] = epsilon_greedy_samples["no_neighbors"].astype(int)
epsilon_greedy_samples["epsilon"] = epsilon_greedy_samples["epsilon"].astype(float)

# Save the samples to a csv file and add index column
epsilon_greedy_samples.to_csv(
    os.path.join(samples_folder, "epsilon_greedy_samples.csv"),
    index_label="index",
)

# Define UCB-U parameter ranges and generate samples
ucb_u_samples, keys = sampling.sobol_prefix_samples(
    UCB_U_PARAM_RANGE, N_SAMPLES, seed=2716,
)

# Transform to a pandas df
ucb_u_samples = pd.DataFrame(
    ucb_u_samples, columns=keys
)
# Correct the data types
ucb_u_samples["sliding_window"] = ucb_u_samples["sliding_window"].astype(int)
ucb_u_samples["no_actions"] = ucb_u_samples["no_actions"].astype(int)
ucb_u_samples["no_neighbors"] = ucb_u_samples["no_neighbors"].astype(int)
ucb_u_samples["alpha"] = ucb_u_samples["alpha"].astype(float)

# Save the UCB-U samples to a csv file and add index column
ucb_u_samples.to_csv(
    os.path.join(samples_folder, "ucb_u_samples.csv"),
    index_label="index",
)

# Define UTS parameter ranges and generate samples
uts_samples, keys = sampling.sobol_prefix_samples(
    UTS_PARAM_RANGE, N_SAMPLES, seed=2716,
)

# Transform to a pandas df
uts_samples = pd.DataFrame(
    uts_samples, columns=keys
)
# Correct the data types
uts_samples["sliding_window"] = uts_samples["sliding_window"].astype(int)
uts_samples["no_actions"] = uts_samples["no_actions"].astype(int)
uts_samples["no_neighbors"] = uts_samples["no_neighbors"].astype(int)
uts_samples["std"] = uts_samples["std"].astype(float)

# Save the UTS samples to a csv file and add index column
uts_samples.to_csv(
    os.path.join(samples_folder, "uts_samples.csv"),
    index_label="index",
)

# Save the metadata
metadata = {
    "epsilon_greedy_param_range": EPS_GREEDY_PARAM_RANGE,
    "ucb_u_param_range": UCB_U_PARAM_RANGE,
    "uts_param_range": UTS_PARAM_RANGE,
    "n_samples": N_SAMPLES,
    "seed": SEED,
}

with open(os.path.join(samples_folder, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)    


# Plot the samples to visualize the parameter space

plt.figure(figsize=(10, 6))
plt.scatter(
    epsilon_greedy_samples["sliding_window"],
    epsilon_greedy_samples["epsilon"],
    label="Epsilon-Greedy Samples",
    alpha=0.5,
    s=10,
)
plt.xlabel('Sliding Window')
plt.ylabel('Epsilon')
plt.title('Epsilon-Greedy Parameter Samples')
plt.legend()
plt.savefig(os.path.join(samples_folder, "epsilon_greedy_samples.png"))
