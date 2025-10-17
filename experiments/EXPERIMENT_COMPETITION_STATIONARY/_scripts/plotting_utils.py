import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def gaussian_smooth_2d(arr: np.ndarray, sigma: float = 1.0, truncate: float = 3.0) -> np.ndarray:
    g = gaussian_kernel1d(sigma, truncate)
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), axis=1, arr=arr)
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), axis=0, arr=smoothed)
    return smoothed

def plot_smoothed_hist2d(
    x: np.ndarray, 
    y: np.ndarray, 
    bins=100, 
    hist_range=None, 
    sigma: float = 1.0, 
    truncate: float = 3.0, 
    density: bool = False,
    title: str | None = None,
    mark_line: float | None = None,
):
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=hist_range, density=density)
    H_sm = gaussian_smooth_2d(H, sigma=sigma, truncate=truncate)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.figure()
    plt.imshow(H_sm.T, origin="lower", aspect="auto", extent=extent)
    if mark_line is not None:
        plt.axhline(y=mark_line, color='r', linestyle='--')
        plt.axvline(x=mark_line, color='r', linestyle='--')

    plt.xlabel("x")
    plt.ylabel("y")
    if title is None:
        title = f"Smoothed 2D histogram (bins={bins}, sigma={sigma})"
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("density" if density else "count")
    return H_sm, xedges, yedges