"""Cleaned analysis script for ExplorationCollusion experiment."""

import os
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt

from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay

from analysis_combine_noise import produce_combined_regret_dataframe


agent_name_dict = {
    "epsilon_greedy": {
        "parameters": [
            "epsilon",
            "no_actions",
            "no_neighbors",
            "sliding_window",
        ],
        "feature_types": [float, int, int, int],
        "feature_maxs": [1.0, 200, 20, 100]
    },
    "ucb_u": {
        "parameters": [
            "alpha",
            "no_actions",
            "no_neighbors",
            "sliding_window",
        ],
        "feature_types": [float, int, int, int],
        "feature_maxs": [4.0, 200, 20, 100]
    },
    "uts": {
        "parameters": [
            "std",
            "no_actions",
            "no_neighbors",
            "sliding_window",
        ],
        "feature_types": [float, int, int, int],
        "feature_maxs": [0.2, 200, 20, 100]
    },
}


def load_samples(samples_filepath):
    if not os.path.exists(samples_filepath):
        raise FileNotFoundError(f"Parameter file not found: {samples_filepath}")
    return pd.read_csv(samples_filepath, index_col=0)


def load_evaluation_results(results_filepath):
    if not os.path.exists(results_filepath):
        raise FileNotFoundError(f"Results file not found: {results_filepath}")
    return pd.read_csv(results_filepath, index_col=0)


def relabel_axis_ticks(ax, scaler, feature_idx, type, axis="x"):
    """Relabel PDP axis from scaled units back to original (integer) values."""
    # Get tick labels
    if axis == "x":
        ticks = ax.get_xticks()
    else:
        ticks = ax.get_yticks()
    # Inverse transform ticks
    original_ticks = ticks * scaler.scale_[feature_idx]
    if type is int:
        original_ticks = np.round(original_ticks).astype(int)
    else:
        original_ticks = np.round(original_ticks, 2)
    # Set new tick labels
    if axis == "x":
        ax.set_xticklabels(original_ticks)
    else:
        ax.set_yticklabels(original_ticks)


if __name__ == "__main__":
    # DEFINITION OF ANALYSIS PARAMETERS
    agent_name = "uts"  # Change to "epsilon_greedy" for Epsilon-Greedy agent
    case = "average"
    method = "squared"
    no_samples = 2056  # Number of samples to evaluate

    agent_info = agent_name_dict[agent_name]
    agent_feature_types = agent_info["feature_types"]
    agent_feature_maxs = agent_info["feature_maxs"]

    results_df = produce_combined_regret_dataframe(agent_name, case, method, no_samples)

    parameter_list = agent_name_dict[agent_name]["parameters"]
    parameter_df = results_df[parameter_list]

    scaler = MaxAbsScaler()
    scaler.fit(parameter_df.values)
    # Override with your own max values (one per feature)
    custom_max = np.array(agent_feature_maxs)  # example max abs values per feature
    scaler.scale_ = custom_max.astype(float)
    scaler.max_abs_ = custom_max.astype(float) 

    scaled_samples = scaler.transform(parameter_df.values)

    unscaled_samples = parameter_df.values
    values = results_df["average_regret"].values

    # Use HistGradientBoostingRegressor instead of GaussianProcessRegressor
    hgb = HistGradientBoostingRegressor(
        max_depth=5,  # can tune depth for smoothness
        learning_rate=0.1,  # smaller = smoother, more regularization
        max_iter=500,  # number of boosting iterations
        l2_regularization=1.0,  # helps handle noise
        random_state=42,
    )
    try:
        hgb = joblib.load("hgb_model.pkl")
    except FileNotFoundError:
        hgb.fit(scaled_samples[:no_samples], values[:no_samples])
        # joblib.dump(hgb, "hgb_model.pkl")

    # pipe = make_pipeline(scaler, hgb)
    fig, ax = plt.subplots(4, 4, figsize=(12, 8))
    features_1d = [0, 1, 2, 3]
    for i, feature in enumerate(features_1d):
        disp = PartialDependenceDisplay.from_estimator(
            hgb,
            scaled_samples,
            features=[feature],
            feature_names=parameter_df.columns.tolist(),
            kind="average",
            method="recursion",
            grid_resolution=20,
            ax=ax[i, i],
        )
        feature_type = agent_feature_types[i]
        # Relabel x-axis in original integer units
        relabel_axis_ticks(disp.axes_[0,0], scaler, feature, feature_type, axis="x")
        
    features_2d = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for feature1, feature2 in features_2d:
        disp = PartialDependenceDisplay.from_estimator(
            hgb,
            scaled_samples[:no_samples],
            features=[(feature1, feature2)],
            feature_names=parameter_df.columns.tolist(),
            kind="average",
            method="recursion",
            grid_resolution=20,
            ax=ax[feature2, feature1],
        )

        feature1_type = agent_feature_types[feature1]
        feature2_type = agent_feature_types[feature2]
        relabel_axis_ticks(disp.axes_[0,0], scaler, feature1, feature1_type, axis="x")
        relabel_axis_ticks(disp.axes_[0,0], scaler, feature2, feature2_type, axis="y")

    plt.tight_layout()
    output_dir = "../pgp_analysis_contabo/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"pdp_hgb_{agent_name}_{case}_{method}.png"
    )
    plt.savefig(output_path)
    plt.show()