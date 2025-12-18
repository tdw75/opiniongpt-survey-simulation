import os

import pandas as pd

from src.analysis.visualisations import plot_model_metric_comparison
from src.demographics.config import subgroups, dimensions, categories
from src.simulation.utils import create_subdirectory

models = ["opinion_gpt", "persona", "base"]


def main(directory: str = "../data_files"):
    simulation_directory = os.path.join(
        directory, "results", "simulation-500-0_9-unconstrained"
    )
    read_directory = create_subdirectory(simulation_directory, "metrics")
    save_directory = create_subdirectory(simulation_directory, "graphs")

    for grouping in ["subgroup", "dimension", "category"]:
        for metric in ["variances", "misalignment"]:
            means_df = load_and_calculate_metric_means(
                os.path.join(read_directory, f"{grouping}-{metric}.csv"), grouping
            )
            print(means_df)
            plt = plot_model_metric_comparison(
                means_df,
                save_directory=save_directory,
                **METRIC_CONFIG[metric],
                **GROUPING_CONFIG[grouping],
            )


def load_and_calculate_metric_means(csv_path: str, grouping: str) -> pd.DataFrame:
    """
    Loads a CSV and computes the mean for each column.
    """
    df = pd.read_csv(csv_path, index_col=0)
    means = df.groupby(["model", "group"], as_index=False).agg({"value": "mean"})
    means_pivot = means.pivot(index="group", columns="model", values="value").round(4)
    means_pivot = means_pivot.reindex(INDEX_ORDER[grouping])
    cols = models + ["true"] if "true" in means_pivot.columns else models
    return means_pivot[cols]


INDEX_ORDER = {
    "subgroup": list(subgroups.keys()),
    "dimension": list(dimensions.keys()),
    "category": categories,
}


METRIC_CONFIG = {
    "variances": {"metric_name": "Response Variance", "xmax": 0.15},
    "misalignment": {"metric_name": "Misalignment", "xmax": 0.28},
}

GROUPING_CONFIG = {
    "subgroup": {"grouping": "subgroup", "subplot_scale": 0.4},
    "category": {"grouping": "category", "subplot_scale": 0.45},
    "dimension": {"grouping": "dimension", "subplot_scale": 0.55},
}


if __name__ == "__main__":
    main()
