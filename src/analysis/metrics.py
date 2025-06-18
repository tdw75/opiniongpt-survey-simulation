import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from src.analysis.responses import get_response_distribution, FrequencyDist
from src.data.variables import ResponseMap


def calculate_jensen_shannon(
    model_responses: dict[str, pd.Series],
    true_responses: dict[str, pd.Series],
    response_maps: dict[str, ResponseMap],
    **kwargs
) -> dict[str, float]:

    all_qnums, model_dists, true_dists = _prepare_distributions(
        model_responses, true_responses, response_maps, **kwargs
    )

    distances = {}
    for qnum in all_qnums:
        _, model_weights, true_weights = get_sorted_support_and_obs(
            model_dists[qnum], true_dists[qnum]
        )
        distances[qnum] = jensenshannon(model_weights, true_weights)
    return distances


def calculate_wasserstein(
    model_responses: dict[str, pd.Series],
    true_responses: dict[str, pd.Series],
    response_maps: dict[str, ResponseMap],
    **kwargs
) -> dict[str, float]:

    all_qnums, model_dists, true_dists = _prepare_distributions(
        model_responses, true_responses, response_maps, **kwargs
    )
    distances = {}

    for qnum in all_qnums:
        support, model_weights, true_weights = get_sorted_support_and_obs(
            model_dists[qnum], true_dists[qnum]
        )
        distance = wasserstein_distance(
            support, support, u_weights=model_weights, v_weights=true_weights
        )
        distances[qnum] = normalise_wasserstein(distance, support)
    return distances


def _prepare_distributions(
    model_responses: dict[str, pd.Series],
    true_responses: dict[str, pd.Series],
    response_maps: dict[str, ResponseMap],
    **kwargs
) -> tuple:
    default_kwargs = dict(is_normalize=True, is_include_invalid=False)
    kwargs = {**default_kwargs, **kwargs}

    model_dists = get_response_distribution(model_responses, response_maps, **kwargs)
    true_dists = get_response_distribution(true_responses, response_maps, **kwargs)
    all_qnums = set(model_dists.keys()).union(true_dists.keys())

    return all_qnums, model_dists, true_dists


def get_sorted_support_and_obs(a: FrequencyDist, b: FrequencyDist) -> tuple:
    support = sorted(a.keys())
    weights_a = [a[k] for k in support]
    weights_b = [b[k] for k in support]
    return support, weights_a, weights_b


def normalise_wasserstein(distance: float, support: list[int]) -> float:
    norm_factor = max(support) - min(support)
    if norm_factor > 0:
        return distance / norm_factor
    else:
        return 0
