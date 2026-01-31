import os
import sys

import fire
import pandas as pd

print(sys.path)
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())


from src.analysis.visualisations import (
    plot_model_metric_comparison,
    plot_model_metric_comparison_stacked,
)
from src.demographics.config import subgroups, dimensions, categories
from src.simulation.utils import create_subdirectory

models = ["opinion_gpt", "persona", "base"]


def main(simulation_name: str, directory: str = "../data_files"):
    simulation_directory = os.path.join(directory, "results", simulation_name)
    read_directory = create_subdirectory(simulation_directory, "metrics")
    save_directory = create_subdirectory(simulation_directory, "graphs")

    for metric in ["variances", "misalignment"]:
        sg_means = load_and_calculate_metric_means(
            os.path.join(read_directory, f"subgroup-{metric}.csv"), "subgroup"
        )
        dim_means = load_and_calculate_metric_means(
            os.path.join(read_directory, f"dimension-{metric}.csv"), "dimension"
        )
        plot_model_metric_comparison_stacked(
            sg_means,
            dim_means,
            save_directory=save_directory,
            subplot_scale=0.35,
            **METRIC_CONFIG[metric],
        )

        cat_means = load_and_calculate_metric_means(
            os.path.join(read_directory, f"category-{metric}.csv"), "category"
        )
        plot_model_metric_comparison(
            cat_means,
            save_directory=save_directory,
            **METRIC_CONFIG[metric],
            **GROUPING_CONFIG["category"],
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
    "misalignment": {"metric_name": "Dissimilarity", "xmax": 0.28},
}

GROUPING_CONFIG = {
    "subgroup": {"grouping": "subgroup", "subplot_scale": 0.4},
    "category": {"grouping": "category", "subplot_scale": 0.45},
    "dimension": {"grouping": "dimension", "subplot_scale": 0.55},
}


if __name__ == "__main__":
    fire.Fire(main)
