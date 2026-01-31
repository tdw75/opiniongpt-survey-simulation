import json
import os
import time
from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd

from src.analysis.metrics import (
    calculate_misalignment,
    calculate_wasserstein,
    calculate_variance,
    prepare_distributions_single,
)
from src.analysis.responses import (
    sort_by_qnum_index,
    get_true_responses_for_subgroup,
    get_model_responses_for_subgroup,
)
from src.analysis.visualisations import (
    plot_distance_heatmap,
)
from src.data.variables import (
    QNum,
    ResponseMap,
    remap_response_maps,
    non_ordinal_qnums,
)
from src.demographics.base import BaseSubGroup
from src.demographics.config import (
    dimensions,
    subgroups,
    categories,
    category_to_question,
)
from src.simulation.models import ModelName, AdapterName
from src.simulation.utils import key_as_int, create_subdirectory, save_latex_table

steered_models = ["opinion_gpt", "persona"]
all_models = steered_models + ["base"]
DataDict = dict[
    AdapterName, dict[ModelName, pd.DataFrame]
]  # subgroup -> model -> DataFrame


# todo: slow and monolithic, refactor
# todo: split into separate modules


def main(filename: str, directory: str = "../data_files"):
    start = time.time()

    simulation_directory = os.path.join(directory, "results", filename)
    sim = pd.read_csv(
        os.path.join(simulation_directory, f"{filename}-clean.csv"), index_col=0
    )
    if "final_response" not in sim.columns:
        sim["final_response"] = sim["response_key"]
    sim = sim.loc[sim["number"] != "Q215"]  # not asked in USA
    true = pd.read_csv(
        os.path.join(directory, "WV7/WVS_Cross-National_Wave_7_csv_v6_0.csv"),
        index_col=0,
    )

    with open(
        os.path.join(directory, "variables/response_map_original.json"), "r"
    ) as f1:
        response_map = key_as_int(json.load(f1))
        response_map = remap_response_maps(response_map)
        response_map = {k: v for k, v in response_map.items() if k != "Q215"}
    all_qnums = list(response_map.keys())

    sim["subgroup"].fillna("none", inplace=True)
    base = get_model_responses(sim[sim["subgroup"] == "none"], all_qnums)
    print(f"Loaded data, {time.time() - start} seconds")
    subgroup_data: DataDict = {
        n: collate_subgroup_data(true, sim, base, s, all_qnums)
        for n, s in subgroups.items()
    }
    dimension_data: DataDict = {
        n: collate_subgroup_data(true, sim, base, s, all_qnums)
        for n, s in dimensions.items()
    }
    category_data = aggregate_by_category(subgroup_data, base, true)
    print(f"Aggregated data, {time.time() - start} seconds")
    metrics_directory = create_subdirectory(simulation_directory, "metrics")
    data_directory = create_subdirectory(simulation_directory, "data")
    graph_directory = create_subdirectory(simulation_directory, "graphs")
    latex_directory = create_subdirectory(simulation_directory, "latex")

    persist_data_dict(subgroup_data, data_directory, "subgroup")
    persist_data_dict(dimension_data, data_directory, "dimension")

    generate_modal_collapse_analysis(
        subgroup_data, base, metrics_directory, latex_directory
    )
    print(f"Finished modal collapse analysis, {time.time() - start} seconds")
    generate_invalid_response_analysis(
        subgroup_data, metrics_directory, latex_directory
    )
    print(f"Finished invalid response analysis, {time.time() - start} seconds")
    generate_invalid_response_analysis(
        category_data, metrics_directory, latex_directory
    )

    data_dict_map = {
        "subgroup": subgroup_data,
        "dimension": dimension_data,
        "category": category_data,
    }

    for g, dd in data_dict_map.items():
        save_response_distributions(
            dd, create_subdirectory(simulation_directory, "data"), response_map, g
        )

    for grouping, data_dict in data_dict_map.items():

        generate_model_comparison_metrics(
            data_dict, response_map, metrics_directory, grouping
        )
        print(
            f"Finished model comparison metrics for {grouping}, {time.time() - start} seconds"
        )
        if grouping != "category":
            generate_cross_comparison(
                data_dict, response_map, graph_directory, grouping
            )


def generate_model_comparison_metrics(
    data_dict: DataDict,
    response_map: dict[QNum, ResponseMap],
    metric_directory: str,
    grouping: str,
):

    misalignment = {
        n: get_metric(d, calculate_misalignment, response_map)
        for n, d in data_dict.items()
    }
    variances = {n: get_variance(d, response_map) for n, d in data_dict.items()}

    flatten_to_df_long(variances).to_csv(
        os.path.join(metric_directory, f"{grouping}-variances.csv")
    )
    flatten_to_df_long(misalignment).to_csv(
        os.path.join(metric_directory, f"{grouping}-misalignment.csv")
    )


def generate_cross_comparison(
    data_dict: DataDict,
    response_map: dict[QNum, ResponseMap],
    graph_directory: str,
    grouping: str,
):

    metric_map = {
        "Wasserstein Distance": (calculate_wasserstein, "Blues"),
        "Misalignment": (calculate_misalignment, "Blues"),
    }
    for name, (fn, cmap) in metric_map.items():
        cross = get_cross_distance(data_dict, fn, response_map)
        plot_distance_heatmap(
            cross,
            name,
            cmap=cmap,
            save_directory=graph_directory,
            grouping=grouping,
        )


def save_response_distributions(
    data_dict: DataDict,
    data_directory: str,
    response_map: dict[QNum, ResponseMap],
    grouping: str,
):
    dists = {
        n: get_response_distributions(d, response_map) for n, d in data_dict.items()
    }
    with open(
        os.path.join(data_directory, f"{grouping}-response-dists.json"), "w"
    ) as f:
        json.dump(dists, f)


def generate_modal_collapse_analysis(
    data_dict: DataDict,
    base: pd.DataFrame,
    metrics_directory: str,
    latex_directory: str,
):
    with open(os.path.join(metrics_directory, "degenerate-dists.json"), "w") as f1:
        deg, counts = find_degenerate_dists(data_dict, base)
        json.dump(deg, f1)

    with open(os.path.join(metrics_directory, "degenerate-qnums.json"), "w") as f2:
        qnums = find_degenerate_questions(deg)
        json.dump(qnums, f2)

    save_latex_table(
        counts.T,
        latex_directory,
        "degenerate-counts-table.tex",
        # label="tab:degenerate-counts",
        # caption="Number of questions with only a single response per model and subgroup",
    )


def generate_invalid_response_analysis(
    subgroup_data: DataDict, metrics_directory: str, latex_directory: str
):
    def _get_invalid_means(df: pd.DataFrame) -> pd.Series:
        return (df == -1).mean()

    def _get_invalid_by_qnum(df: pd.DataFrame) -> dict[QNum, float]:
        return _get_invalid_means(df)[lambda x: x > 0.1].to_dict()

    invalid_counts = {}
    invalid_totals = {}
    for sg, dd in subgroup_data.items():
        invalid_counts[sg] = {m: _get_invalid_by_qnum(dd[m]) for m in dd.keys()}
        invalid_totals[sg] = {m: _get_invalid_means(dd[m]).mean() for m in dd.keys()}

    invalid_totals = pd.DataFrame(invalid_totals).T
    save_latex_table(
        invalid_totals,
        latex_directory,
        "invalid-counts-total.tex",
        float_format="%.3f",
    )
    invalid_totals.to_csv(os.path.join(metrics_directory, "invalid-counts-total.csv"))

    with open(
        os.path.join(metrics_directory, "invalid-counts-question.json"), "w"
    ) as f:
        json.dump(invalid_counts, f)

    common_qnums = {m: Counter() for m in steered_models + ["true", "base"]}
    for sg, models in invalid_counts.items():
        for m, qnums in models.items():
            common_qnums[m].update(qnums.keys())

    for m, counts in common_qnums.items():
        common_qnums[m] = {i: counts[i] for i in sorted(counts.keys())}

    with open(
        os.path.join(metrics_directory, "common-invalid-questions.json"), "w"
    ) as f:
        json.dump(common_qnums, f)


def collate_subgroup_data(
    df_true: pd.DataFrame,
    df_sim: pd.DataFrame,
    df_base: pd.DataFrame,
    subgroup: type[BaseSubGroup] | list[type[BaseSubGroup]],
    qnums: list[QNum],
) -> dict[str, pd.DataFrame]:
    # qnums columns, obs rows

    return {
        "true": pd.DataFrame(get_true_responses_for_subgroup(df_true, subgroup, qnums)),
        "opinion_gpt": pd.DataFrame(
            get_model_responses_for_subgroup(df_sim[df_sim["is_lora"]], subgroup, qnums)
        ),
        "persona": pd.DataFrame(
            get_model_responses_for_subgroup(
                df_sim[~df_sim["is_lora"]], subgroup, qnums
            )
        ),
        "base": df_base,
    }


def get_question_means(
    subgroup_dict: DataDict,
    model_name: str,
    diameters: pd.Series,
    minimums: pd.Series,
    filter_val: str = None,
) -> pd.DataFrame:
    """
    Calculate the mean response for each question across all subgroups, normalised by the response diameter.
    """
    means = {}
    for sg, dd in subgroup_dict.items():
        df = dd[model_name].drop(
            columns=non_ordinal_qnums() + ["weight"], errors="ignore"
        )
        if filter_val is not None:
            df = df.filter(like=filter_val)
        sg_means = df.replace(-1, np.nan).mean().round(2)  # drop invalid responses
        means[sg] = sort_by_qnum_index(sg_means) - sort_by_qnum_index(minimums)
        means[sg] = means[sg] / sort_by_qnum_index(diameters)

    return pd.DataFrame(means)


def get_category_means(question_means: pd.DataFrame) -> pd.DataFrame:
    cat_means = {}
    for cat, qnums in category_to_question.items():
        df_loop = question_means.filter(items=qnums, axis=0)
        cat_means[cat] = df_loop.mean().round(2)

    return pd.DataFrame(cat_means).T


def get_model_responses(df_sim: pd.DataFrame, qnums: list[QNum]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            qnum: df_sim.loc[df_sim["number"] == qnum, "final_response"].values
            for qnum in qnums
        }
    )


def find_degenerate_dists(data: dict, base: pd.DataFrame) -> tuple[dict, pd.DataFrame]:

    def _find_single(df: pd.DataFrame) -> list[str]:
        return [
            qnum for qnum in df.columns if df.loc[df[qnum] >= 0, qnum].nunique() == 1
        ]

    degenerate_dists, counts = {"base": _find_single(base)}, {}
    for s, _ in data.items():
        degenerate_dists[s] = {m: _find_single(data[s][m]) for m in steered_models}
        counts[s] = {
            m: pd.Series(degenerate_dists[s][m]).count()
            for m in degenerate_dists[s].keys()
        }

    return degenerate_dists, pd.DataFrame(counts)


def find_degenerate_questions(
    degenerate_dists: dict[ModelName : dict[AdapterName, list[QNum]]],
) -> dict[QNum, int]:
    """
    Find questions that have degenerate distributions across all models.
    """
    degenerate_questions = Counter()
    for adapter, models in degenerate_dists.items():
        if adapter == "base":
            degenerate_questions.update(models)
        else:
            for qnums in models.values():
                degenerate_questions.update(qnums)

    return degenerate_questions


def get_response_distributions(
    dfs: dict[str, pd.DataFrame], response_map: dict[QNum, ResponseMap]
):
    return {
        "opinion_gpt": prepare_distributions_single(dfs["opinion_gpt"], response_map),
        "persona": prepare_distributions_single(dfs["persona"], response_map),
        "base": prepare_distributions_single(dfs["base"], response_map),
        "true": prepare_distributions_single(dfs["true"], response_map),
    }


def get_metric(
    dfs: dict[str, pd.DataFrame], metric_fn, response_map: dict[QNum, ResponseMap]
) -> dict[str, pd.Series]:
    return {
        "opinion_gpt": pd.Series(
            metric_fn(dfs["opinion_gpt"], dfs["true"], response_map)
        ),
        "persona": pd.Series(metric_fn(dfs["persona"], dfs["true"], response_map)),
        "base": pd.Series(metric_fn(dfs["base"], dfs["true"], response_map)),
    }


def flatten_to_df(metric_results: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame()
    for sg, metrics in metric_results.items():
        for model, vals in metrics.items():
            df[f"{sg}_{model}"] = vals
    return sort_by_qnum_index(df.round(4))


def flatten_to_df_long(metric_results: dict[str, dict]) -> pd.DataFrame:
    records = []
    for sg, metrics in metric_results.items():
        for model, vals in metrics.items():
            for qnum, value in pd.Series(vals).items():
                records.append(
                    {"number": qnum, "group": sg, "model": model, "value": value}
                )
    cols = ["number", "group", "model", "value"]
    df = pd.DataFrame(records)
    return df[cols].sort_values(cols[1:]).reset_index(drop=True)


def get_variance(
    dfs: dict[str, pd.DataFrame], response_map: dict[QNum, ResponseMap]
) -> dict[str, pd.Series]:
    return {
        "opinion_gpt": pd.Series(calculate_variance(dfs["opinion_gpt"], response_map)),
        "persona": pd.Series(calculate_variance(dfs["persona"], response_map)),
        "true": pd.Series(calculate_variance(dfs["true"], response_map)),
        "base": pd.Series(calculate_variance(dfs["base"], response_map)),
    }


def filter_qnums(df: pd.DataFrame, qnums: list[QNum]) -> pd.DataFrame:
    df = df.copy()
    return df.drop(columns=qnums, errors="ignore")


def get_metric_means(metric_results: dict, keys: list[str] = None) -> pd.DataFrame:
    means = {}
    for sg in metric_results.keys():
        means[sg] = {m: metric_results[sg][m].mean() for m in keys or all_models}
    return pd.DataFrame(means).T.round(4)


def get_cross_distance(
    data_dict: dict,
    metric_fn: Callable,
    response_map: dict[QNum, ResponseMap],
    data_name: str = "true",
) -> pd.DataFrame:
    cross = {}
    data_dict = data_dict.copy()
    # data_dict["Base Phi 3"] = {data_name: base}
    for s1, d1 in data_dict.items():
        cross[s1] = {}
        for s2, d2 in data_dict.items():
            cross[s1][s2] = pd.Series(
                metric_fn(d1[data_name], d2[data_name], response_map)
            ).mean()

    return pd.DataFrame(cross).T.round(4)


# def aggregate_by_dimension(data_dict: DataDict):
#     for name, data in data_dict.items():



def aggregate_by_category(
    data_dict: DataDict, base: pd.DataFrame, true: pd.DataFrame
) -> dict:

    all_qnums = set(base.columns)
    cat_dict = {c: {m: [] for m in steered_models + ["true"]} for c in categories}

    for cat, qnums in category_to_question.items():
        qnums = all_qnums.intersection(qnums)
        for sg, sources in data_dict.items():
            for model, df in sources.items():
                if model == "base":
                    continue
                elif model == "true":
                    df = true
                df_loop = df.filter(items=qnums)
                df_loop.index = [f"{sg}_{i}" for i in df_loop.index]
                cat_dict[cat][model].append(df_loop)
            cat_dict[cat]["base"] = [base.filter(items=qnums)]

    cat_dict = {
        c: {m: pd.concat(dfs) for m, dfs in models.items()}
        for c, models in cat_dict.items()
    }
    return cat_dict


def persist_data_dict(data_dict: DataDict, directory: str, grouping: str):
    for sg, models in data_dict.items():
        for model, df in models.items():
            if model != "base":
                df.to_csv(
                    os.path.join(directory, f"{grouping}-{model}-{sg}-responses.csv")
                )
    data_dict[sg]["base"].to_csv(
        os.path.join(directory, f"{grouping}-base-responses.csv")
    )


if __name__ == "__main__":
    main(filename=f"simulation-500-0_9-unconstrained")
