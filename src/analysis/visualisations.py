import os
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from src.simulation.models import ModelName

TICK_FONT_SIZE = 14


def plot_model_metric_comparison(
    df: pd.DataFrame,
    metric_name: str = "Value",
    save_directory: str = None,
    grouping: str = "",
):
    df = df.rename(columns=RENAME_MAP, errors="ignore")
    groups = df.index
    variants = df.columns
    fig, ax = plt.subplots(figsize=(8, len(groups) * 0.7))

    markers = ["o", "s", "^", "D"]  # circle, square, triangle, diamond
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red
    jitter_amount = 0.035

    for i, group in enumerate(groups):
        x = df.loc[group].values

        for j, (xi, variant) in enumerate(zip(x, variants)):
            y = i + (j - (len(variants) - 1) / 2) * jitter_amount
            ax.plot(
                xi,
                y,
                marker=markers[j],
                color=colors[j],
                markersize=7,
                markeredgecolor="black",
                markeredgewidth=0.5,
                linestyle="None",
                label=variant if i == 0 else "",
            )  # Only label once for legend

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(_wrap_labels(reformat_index(groups), width=22))
    ax.margins(y=0.02)
    ax.set_ylim(-0.5, len(groups) - 0.5)

    max_x = df.values.max() * 1.05
    if metric_name == "Jensen Shannon Distance":
        xmax = 0.36
    elif metric_name in ["Wasserstein Distance", "Misalignment"]:
        xmax = 0.37
    elif metric_name == "Response Variance":
        xmax = 0.17
    elif metric_name == "Response Standard Deviation":
        xmax = 0.25
    else:
        xmax = max_x
    ax.set_xlim(-0.005, max(max_x, xmax))

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.xaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.legend(
        title="Model", loc="lower left" if metric_name == "Misalignment" else "best"
    )
    ax.set_xlabel(f"Mean {metric_name}")

    plt.tight_layout()

    if save_directory:
        save_path = os.path.join(
            save_directory, f"{metric_name.lower()} model comparison {grouping}.png"
        )
        plt.savefig(save_path)


def plot_distance_heatmap(
    distance_matrix: pd.DataFrame,
    metric_name: str = "Value",
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    annot: bool = True,
    fmt: str = ".3f",
    save_directory: str = None,
    grouping: str = "",
):
    """
    Create a heatmap visualization for a distance matrix.

    Parameters:
    -----------
    distance_matrix : pandas.DataFrame or numpy.array
        Square matrix containing distance values between pairs
    title : str, default "Distance Matrix Heatmap"
        Title for the heatmap
    figsize : tuple, default (10, 8)
        Figure size (width, height)
    cmap : str, default 'viridis'
        Colormap for the heatmap. Good options for distance:
        'viridis', 'plasma', 'Blues', 'Reds', 'YlOrRd'
    annot : bool, default True
        Whether to annotate cells with values
    fmt : str, default '.2f'
        Format for annotations
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """

    if isinstance(distance_matrix, np.ndarray):
        distance_matrix = pd.DataFrame(distance_matrix)
    distance_matrix.index = reformat_index(distance_matrix.index)
    distance_matrix.columns = reformat_index(distance_matrix.columns)

    min_distance, max_distance = 0, distance_matrix.max().max()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(distance_matrix, dtype=bool))
    np.fill_diagonal(mask, False)

    sns.heatmap(
        distance_matrix,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=min_distance,
        vmax=max_distance,
        square=True,  # Makes cells square-shaped
        cbar_kws={"label": f"{metric_name}"},
        ax=ax,
    )

    plt.xticks(rotation=45, ha="right", fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.tight_layout()

    if save_directory:
        save_path = os.path.join(
            save_directory, f"{metric_name.lower()} cross comparison {grouping}.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_boxplots(data_dict: dict, model: str, save_directory: str = None):

    labels = [
        "Conser-vative" if k == "Conservative" else k
        for k in reformat_index(data_dict.keys())
    ]
    mean_of_means = np.mean([sg[model].mean() for sg in data_dict.values()])
    print(f"Mean {model}: {mean_of_means}")
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        [sg[model] for sg in data_dict.values()],
        labels=_wrap_labels(labels, width=8),
    )
    plt.ylabel("Values")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.axhline(y=0)
    plt.axhline(
        y=float(mean_of_means),
        linestyle="--",
        label=f"Mean over all models: {mean_of_means:.3f}",
    )
    plt.ylim(-1, 1)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(rotation=0, fontsize=TICK_FONT_SIZE)
    plt.legend()
    if save_directory:
        save_path = os.path.join(save_directory, f"{model.lower()} boxplot.png")
        plt.savefig(save_path)


def _upper_triangle_values(df: pd.DataFrame) -> np.ndarray:
    """Return the upper-triangular (k=1) entries of a square matrix as a 1D array."""
    A = np.asarray(df)
    iu = np.triu_indices_from(A, k=1)
    return A[iu]


def _paired_upper_triangle(
    corr_wvs: pd.DataFrame, corr_model: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned upper-triangle vectors (exclude NaNs)."""
    x = _upper_triangle_values(corr_wvs).astype(float)
    y = _upper_triangle_values(corr_model).astype(float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]


def _wrap_labels(labels, width=22):
    return ["\n".join(wrap(label, width)) for label in labels]


def reformat_index(index: pd.Index | list) -> pd.Index:
    if not isinstance(index, pd.Index):
        index = pd.Index(index)
    return index.str.replace("_", " ").str.title()


RENAME_MAP = {
    "opinion_gpt": "OpinionGPT",
    "persona": "Persona Prompting",
    "true": "True Data",
    "base": "Base Phi 3 Mini",
}
