import pandas as pd

from src.demographics.base import BaseSubGroup
from variables import ResponseMap

FrequencyDist = dict[str, int]
FrequencyDistNormalized = dict[str, float]
ResponseSupport = list[int]


def get_true_responses_for_subgroup(
    df: pd.DataFrame, subgroup: type[BaseSubGroup], qnums: list[str]
) -> dict[str, pd.Series]:
    is_subgroup = subgroup.filter_true(df)
    return {qnum: df.loc[is_subgroup, qnum] for qnum in qnums}


def get_model_responses_for_subgroup(
    df: pd.DataFrame, subgroup: type[BaseSubGroup], qnums: list[str]
) -> dict[str, pd.Series]:
    is_subgroup = subgroup.filter_model(df)
    return {
        qnum: df.loc[is_subgroup & (df["number"] == qnum), "response_key"]
        for qnum in qnums
    }


def get_response_distribution(
    responses: dict[str, pd.Series],
    response_maps: dict[str, ResponseMap],
    is_normalize: bool = False,
    is_include_invalid: bool = False,
) -> dict[str, FrequencyDist | FrequencyDistNormalized]:

    dists = {}
    for qnum, observations in responses.items():
        support = list(response_maps[qnum].keys())

        if not is_include_invalid:
            support = [x for x in support if x > -1]
            observations = observations[observations > -1]

        counts = observations.value_counts(normalize=is_normalize, sort=False)
        counts = counts.reindex(support, fill_value=0)
        dists[qnum] = {
            k: float(v) if is_normalize else int(v) for k, v in counts.items()
        }

    return dists
