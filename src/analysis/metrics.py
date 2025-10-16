from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from src.analysis.responses import (
    get_response_distribution,
    FrequencyDist,
    remove_weight_col,
    get_response_distribution_weighted,
    get_support_diameter,
    get_support_minimum,
)
from src.data.variables import ResponseMap, QNum, ordinal_qnums, non_ordinal_qnums


def calculate_jensen_shannon(
    model_responses: pd.DataFrame,
    true_responses: pd.DataFrame,
    response_maps: dict[QNum, ResponseMap],
    **kwargs
) -> dict[QNum, float]:
    return _calculate_distance_metric(
        jensenshannon,
        non_ordinal_qnums(),
        model_responses,
        true_responses,
        response_maps,
        **kwargs
    )


def calculate_total_variation(
    model_responses: pd.DataFrame,
    true_responses: pd.DataFrame,
    response_maps: dict[QNum, ResponseMap],
    **kwargs
) -> dict[QNum, float]:
    return _calculate_distance_metric(
        total_variation_distance,
        non_ordinal_qnums(),
        model_responses,
        true_responses,
        response_maps,
        **kwargs
    )


def calculate_variance(
    responses: pd.DataFrame, response_maps: dict[QNum, ResponseMap], **kwargs
):
    dists = prepare_distributions_single(responses, response_maps, **kwargs)
    qnums = set(dists.keys()).intersection(ordinal_qnums())

    variances = {}
    for qnum in qnums:
        support, weights = get_sorted_support_and_obs_single(dists[qnum])
        mu = np.dot(weights, support)
        variance = np.dot(weights, (support - mu) ** 2)
        variances[qnum] = normalise(variance, support, 2)
    return variances


def _calculate_distance_metric(
    metric_fn: Callable,
    qnum_subset: list[QNum],
    model_responses: pd.DataFrame,
    true_responses: pd.DataFrame,
    response_maps: dict[QNum, ResponseMap],
    **kwargs
):
    model_dists, true_dists = prepare_distributions(
        model_responses, true_responses, response_maps, **kwargs
    )
    qnums = set(model_dists.keys()).intersection(true_dists).intersection(qnum_subset)

    distances = {}
    for qnum in qnums:
        _, model_weights, true_weights = get_sorted_support_and_obs(
            model_dists[qnum], true_dists[qnum]
        )
        distances[qnum] = metric_fn(model_weights, true_weights)
    return distances


def calculate_wasserstein(
    model_responses: pd.DataFrame,
    true_responses: pd.DataFrame,
    response_maps: dict[QNum, ResponseMap],
    **kwargs
) -> dict[QNum, float]:

    model_dists, true_dists = prepare_distributions(
        model_responses, true_responses, response_maps, **kwargs
    )
    qnums = (
        set(true_dists.keys())
        .intersection(model_dists.keys())
        .intersection(ordinal_qnums())
    )
    distances = {}

    for qnum in qnums:
        try:
            support, model_weights, true_weights = get_sorted_support_and_obs(
                model_dists[qnum], true_dists[qnum]
            )
        except KeyError:
            print(qnum)
            support, model_weights, true_weights = get_sorted_support_and_obs(
                model_dists[qnum], true_dists[qnum]
            )
        # Check for valid weights
        if (
            len(model_weights) == 0
            or len(true_weights) == 0
            or not np.isfinite(model_weights).all()
            or not np.isfinite(true_weights).all()
            or np.sum(model_weights) <= 0
            or np.sum(true_weights) <= 0
        ):
            distances[qnum] = np.nan
        else:
            distance = wasserstein_distance(
                support, support, u_weights=model_weights, v_weights=true_weights
            )
            distances[qnum] = normalise(distance, support, 1)
    return distances


def calculate_misalignment(
    model_responses: pd.DataFrame,
    true_responses: pd.DataFrame,
    response_maps: dict[QNum, ResponseMap],
    **kwargs
) -> dict[QNum, float]:
    wasserstein = calculate_wasserstein(
        model_responses, true_responses, response_maps, **kwargs
    )
    total_variation = calculate_total_variation(
        model_responses, true_responses, response_maps, **kwargs
    )
    return {**wasserstein, **total_variation}


def calculate_variance0(
    responses: pd.DataFrame, response_maps: dict[QNum, ResponseMap], **kwargs
) -> dict[QNum, float]:
    # todo: vectorise
    variances = {}
    for qnum in remove_weight_col(responses.columns):
        if qnum in ordinal_qnums():  # only ordinal
            variance = responses.loc[responses[qnum] > -1, qnum].var()
            support = [x for x in response_maps[qnum].keys() if x > -1]
            variances[qnum] = normalise(variance, support, 2)

    return variances


def calculate_weighted_variance(
    responses: pd.DataFrame, response_maps: dict[QNum, ResponseMap]
):

    weights = responses["weight"].to_numpy()[:, None]
    weights_sum = np.sum(weights)
    responses = responses.drop(columns="weight").to_numpy(dtype=float)
    responses = normalise_responses(responses, response_maps)

    mean_w = np.sum(weights * responses, axis=0) / weights_sum
    var_w = np.sum(weights * (responses - mean_w) ** 2, axis=0) / weights_sum
    return var_w


def normalise_responses(
    responses: pd.DataFrame,
    response_maps: dict[QNum, ResponseMap],
    is_just_ordinal: bool,
):
    diameter = get_support_diameter(response_maps, is_just_ordinal)
    mins = get_support_minimum(response_maps, is_just_ordinal)
    if is_just_ordinal:
        cols = [c for c in responses.columns if c in ordinal_qnums()]
        responses = responses[cols]
    responses = responses[responses > -1]
    responses -= mins
    responses /= diameter
    return responses


def calculate_variance1(
    responses: pd.DataFrame, response_maps: dict[QNum, ResponseMap], **kwargs
) -> pd.Series:
    # qnums = [c for c in responses.columns if c in ordinal_qnums()]
    qnums = remove_weight_col(responses.columns)
    responses = normalise_responses(
        responses[qnums], response_maps, is_just_ordinal=True
    )
    return responses.var()
    # for qnum in set(non_ordinal_qnums()).intersection(response_maps.keys()):
    #     variances[qnum] = responses[qnum]


def calculate_mean(
    responses: pd.DataFrame, response_maps: dict[QNum, ResponseMap], **kwargs
) -> pd.Series:
    """Calculate normalised mean for each question in responses DataFrame"""
    return normalise_responses(responses, response_maps, is_just_ordinal=False).mean()


def calculate_difference_in_means(
    model_responses: pd.DataFrame,
    true_responses: pd.DataFrame,
    response_maps: dict[QNum, ResponseMap],
):
    model_means = calculate_mean(model_responses, response_maps)
    true_means = calculate_mean(true_responses, response_maps)
    return model_means - true_means


def prepare_distributions(
    model_responses: pd.DataFrame,
    true_responses: pd.DataFrame,
    response_maps: dict[QNum, ResponseMap],
    **kwargs
) -> tuple:
    default_kwargs = dict(is_normalize=True, is_include_invalid=False)
    kwargs = {**default_kwargs, **kwargs}

    if "weight" in model_responses.columns:
        model_dists = get_response_distribution_weighted(
            model_responses, response_maps, **kwargs
        )
    else:
        model_dists = get_response_distribution(
            model_responses, response_maps, **kwargs
        )

    if "weight" in true_responses.columns:
        true_dists = get_response_distribution_weighted(
            true_responses, response_maps, **kwargs
        )
    else:
        true_dists = get_response_distribution(true_responses, response_maps, **kwargs)

    return model_dists, true_dists


def prepare_distributions_single(
    responses: pd.DataFrame, response_maps: dict[QNum, ResponseMap], **kwargs
) -> dict[QNum, FrequencyDist]:
    default_kwargs = dict(is_normalize=True, is_include_invalid=False)
    kwargs = {**default_kwargs, **kwargs}

    if "weight" in responses.columns:
        dists = get_response_distribution_weighted(responses, response_maps, **kwargs)
    else:
        dists = get_response_distribution(responses, response_maps, **kwargs)

    return dists


def get_sorted_support_and_obs(a: FrequencyDist, b: FrequencyDist) -> tuple:
    support = [x for x in sorted(a.keys()) if x > -1]
    weights_a = np.array([a.get(k, 0) for k in support])
    weights_b = np.array([b.get(k, 0) for k in support])
    return support, weights_a, weights_b


def get_sorted_support_and_obs_single(a: FrequencyDist) -> tuple:
    support = [x for x in sorted(a.keys()) if x > -1]
    weights_a = np.array([a.get(k, 0) for k in support])
    return support, weights_a


def normalise(distance: float | pd.Series, support: list[int], order: int = 1) -> float:
    support = [x for x in support if x > -1]
    diameter = max(support) - min(support)
    if diameter > 0:
        return distance / diameter**order
    else:
        return 0


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Total Variation distance between two distributions.
    Parameters
    ----------
    p, q : array_like
        Probability vectors (must sum to 1).
    """
    if not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1):
        raise ValueError("Inputs must sum to 1.")
    return 0.5 * np.abs(p - q).sum()
