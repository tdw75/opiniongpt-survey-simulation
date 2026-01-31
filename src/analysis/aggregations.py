import os

import pandas as pd

from src.analysis.responses import (
    get_true_responses_for_subgroup,
    get_model_responses_for_subgroup,
)
from src.data.variables import QNum
from src.demographics.base import BaseSubGroup
from src.demographics.config import categories, category_to_question
from src.simulation.models import AdapterName, ModelName

steered_models = ["opinion_gpt", "persona"]
all_models = steered_models + ["base"]
DataDict = dict[
    AdapterName, dict[ModelName, pd.DataFrame]
]  # subgroup -> model -> DataFrame


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


def get_model_responses(df_sim: pd.DataFrame, qnums: list[QNum]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            qnum: df_sim.loc[df_sim["number"] == qnum, "final_response"].values
            for qnum in qnums
        }
    )
