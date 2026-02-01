import os
import sys

import fire
import numpy as np
import pandas as pd

print(sys.path)
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())


from src.analysis.aggregations import all_models
from src.analysis.correlations import (
    lower_bound,
    upper_bound,
    compare_correlation_structures,
)
from src.analysis.responses import (
    get_support_diameter,
    get_support_minimum,
    sort_by_qnum_index,
)
from src.simulation.utils import load_response_maps, load_data_dict, save_latex_table


def main(simulation_name: str, directory: str = "../data_files", random_seed: int = 42):
    np.random.seed(random_seed)
    response_maps = load_response_maps(directory)
    diameters = sort_by_qnum_index(get_support_diameter(response_maps))
    minimums = sort_by_qnum_index(get_support_minimum(response_maps))

    subgroup_data = load_data_dict(
        simulation_name, directory, all_models + ["true"], grouping="subgroup"
    )
    corr_metrics = compare_correlation_structures(
        subgroup_data, diameters, minimums, simulation_name, directory
    )
    print("Correlation metrics computed.")
    lb = lower_bound(simulation_name, directory)
    print("Lower bound computed.")
    ub = upper_bound(diameters, minimums, subgroup_data)
    print("Upper bound computed.")
    for grouping, metrics in corr_metrics.items():
        metrics["Lower"] = lb[0][grouping]
        metrics["Upper"] = ub[0][grouping]

        save_latex_table(
            pd.DataFrame(metrics),
            os.path.join(directory, "results", simulation_name, "latex"),
            f"{grouping}-correlation_metrics.tex",
            float_format="%.3f",
        )


if __name__ == "__main__":
    fire.Fire(main)
