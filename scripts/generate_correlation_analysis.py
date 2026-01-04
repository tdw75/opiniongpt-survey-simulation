import os

import numpy as np
import pandas as pd

from scripts.generate_metrics_summary import (
    all_models,
)
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


# subgroup correlation structure is a lower threshold on performance than complete respondent correlation structure


def main(filename: str, root_dir: str = "../data_files"):
    response_maps = load_response_maps()
    diameters = get_support_diameter(response_maps)
    diameters = sort_by_qnum_index(diameters)
    minimums = get_support_minimum(response_maps)
    minimums = sort_by_qnum_index(minimums)

    subgroup_data = load_data_dict(
        filename, root_dir, all_models + ["true"], grouping="subgroup"
    )
    lb = lower_bound(filename, root_dir)

    ub = upper_bound(diameters, minimums, subgroup_data)
    corr_metrics = compare_correlation_structures(
        subgroup_data, diameters, minimums, filename, root_dir
    )
    for grouping, metrics in corr_metrics.items():
        metrics["Lower"] = lb[0][grouping]
        metrics["Upper"] = ub[0][grouping]

        save_latex_table(
            pd.DataFrame(metrics),
            os.path.join(root_dir, "results", filename, "latex"),
            f"{grouping}-correlation_metrics.tex",
            float_format="%.3f",
        )


if __name__ == "__main__":
    np.random.seed(42)

    main(filename="simulation-500-0_9-unconstrained")
