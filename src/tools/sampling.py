# Define the parameter samples using Sobol sequences
import numpy as np
from scipy.stats import qmc
from typing import Dict, Tuple


def sobol_prefix_samples(
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int,
    *,
    seed: int = 0,
    scramble: bool = True,
) -> np.ndarray:
    """
    Deterministic Sobol samples with prefix stability.
    The first n rows of a larger call are identical to a smaller call (same args).

    Args:
        param_ranges: {name: (low, high)}; columns ordered by sorted(name).
        n_samples: number of rows to return.
        seed: seed for reproducibility (ignored if scramble=False).
        scramble: Owen scrambling for variance reduction.
        balance_power_of_two: if True, draws in 2^m-sized blocks for better balance,
                              then truncates to n_samples (still prefix-stable).
        start_index: starting Sobol index (default 0). Use to “continue” later if desired.

    Returns:
        (n_samples, n_params) array scaled to the provided ranges.
    """
    # Stable column order regardless of dict insertion order
    keys = sorted(param_ranges.keys())
    bounds = np.array([param_ranges[k] for k in keys], dtype=float)
    d = len(keys)

    engine = qmc.Sobol(d=d, scramble=scramble, seed=seed)

    X = engine.random(n_samples)

    lows, highs = bounds[:, 0], bounds[:, 1]
    return lows + X * (highs - lows), keys
