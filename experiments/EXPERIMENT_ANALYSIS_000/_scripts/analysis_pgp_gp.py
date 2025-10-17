"""Cleaned analysis script for ExplorationCollusion experiment."""

import os
import joblib

import matplotlib.pyplot as plt

from sklearn.preprocessing import MaxAbsScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.pipeline import make_pipeline
from sklearn.inspection import PartialDependenceDisplay


from analysis_combine_noise import produce_combined_regret_dataframe

agent_name_dict = {
    "epsilon_greedy": {
        "parameters": [
            "epsilon",
            "no_actions",
            "no_neighbors",
            "sliding_window",
        ]
    },
    "ucb_u": {
        "parameters": [
            "alpha",
            "no_actions",
            "no_neighbors",
            "sliding_window",
        ]
    },
    "uts": {
        "parameters": [
            "std",
            "no_actions",
            "no_neighbors",
            "sliding_window",
        ]
    },
}

# # Renaming map
# rename_map = {
#     "no_actions": "No. actions $K$",
#     "no_neighbors": "No. neighbors $k$",
#     "sliding_window": "Sliding window $\\tau$",
#     "epsilon": "Epsilon $\\varepsilon$",
#     "alpha": "Alpha $\\alpha$",
#     "std": "Std dev $\\sigma$",
# }

# rename_map = {
#     "no_actions": "$K$",
#     "no_neighbors": "$k$",
#     "sliding_window": "$\\tau$",
#     "epsilon": "$\\varepsilon$",
#     "alpha": "$\\alpha$",
#     "std": "$\\sigma$",
# }

rename_map = {
    "no_actions": "Actions $K$",
    "no_neighbors": "Neighbours $k$",
    "sliding_window": "Window $\\tau$",
    "epsilon": "$\\varepsilon$",
    "alpha": "$\\alpha$",
    "std": "SD $\\sigma$",
}



if __name__ == "__main__":
    # DEFINITION OF ANALYSIS PARAMETERS
    agent_name = "epsilon_greedy"  # Change to "epsilon_greedy" for Epsilon-Greedy agent
    case = "nonstationary"
    method = "squared"
    no_samples = 512  # Number of samples to evaluate
    offset = 0

    results_df = produce_combined_regret_dataframe(agent_name, case, method)

    parameter_list = agent_name_dict[agent_name]["parameters"]
    parameter_df = results_df[parameter_list]

    # Rename columns for better readability in plots
    parameter_df = parameter_df.rename(columns=rename_map)

    scaler = MaxAbsScaler()
    scaled_samples = scaler.fit_transform(parameter_df.values)
    unscaled_samples = parameter_df.values
    values = results_df["average_regret"].values

    kernel = C(1.0, (1e-2, 1e2)) * RBF(
        length_scale=[1, 1, 1, 1], length_scale_bounds=(1e-2, 1e2)
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10, normalize_y=True, alpha=1e-3
    )
    try:
        gp = joblib.load("gp_model.pkl")
    except FileNotFoundError:
        gp.fit(scaled_samples[:no_samples], values[:no_samples])
        # joblib.dump(gp, "gp_model.pkl")

    plt.rcParams.update({'font.size': 17})

    pipe = make_pipeline(scaler, gp)
    fig, ax = plt.subplots(4, 4, figsize=(14, 9))
    features_1d = [0, 1, 2, 3]
    for i, feature in enumerate(features_1d):
        pgp_disp = PartialDependenceDisplay.from_estimator(
            pipe,
            unscaled_samples,
            features=[feature],
            feature_names=parameter_df.columns.tolist(),
            kind="average",
            ice_lines_kw={"alpha": 0.2, "color": "gray"},
            grid_resolution=20,
            ax=ax[i, i],
        )
        pgp_disp.axes_[0,0].set_ylabel("")
    features_2d = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for feature1, feature2 in features_2d:
        PartialDependenceDisplay.from_estimator(
            pipe,
            unscaled_samples[:no_samples],
            features=[(feature1, feature2)],
            feature_names=parameter_df.columns.tolist(),
            kind="average",
            grid_resolution=20,
            ax=ax[feature2, feature1],
        )
    # Hide unused subplots
    for i in range(4):
        for j in range(4):
            if i != j and (i, j) not in features_2d:
                ax[j, i].axis("off")

    plt.tight_layout()
    output_dir = "../pgp_analysis/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"pdp_gp_{agent_name}_{case}_{method}.png"
    )
    plt.savefig(output_path)
    plt.show()
