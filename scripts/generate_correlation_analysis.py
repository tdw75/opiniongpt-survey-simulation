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


def main(simulation_name: str, root_dir: str = "../data_files", random_seed: int = 42):
    np.random.seed(random_seed)
    response_maps = load_response_maps()
    diameters = get_support_diameter(response_maps)
    diameters = sort_by_qnum_index(diameters)
    minimums = get_support_minimum(response_maps)
    minimums = sort_by_qnum_index(minimums)

    subgroup_data = load_data_dict(
        simulation_name, root_dir, all_models + ["true"], grouping="subgroup"
    )
    lb = lower_bound(simulation_name, root_dir)

    ub = upper_bound(diameters, minimums, subgroup_data)
    corr_metrics = compare_correlation_structures(
        subgroup_data, diameters, minimums, simulation_name, root_dir
    )
    for grouping, metrics in corr_metrics.items():
        metrics["Lower"] = lb[0][grouping]
        metrics["Upper"] = ub[0][grouping]

        save_latex_table(
            pd.DataFrame(metrics),
            os.path.join(root_dir, "results", simulation_name, "latex"),
            f"{grouping}-correlation_metrics.tex",
            float_format="%.3f",
        )


if __name__ == "__main__":
    fire.Fire(main)

