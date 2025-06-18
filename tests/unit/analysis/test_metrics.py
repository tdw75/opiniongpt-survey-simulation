import pytest
import pandas as pd
import numpy as np

from src.analysis.metrics import calculate_wasserstein, calculate_jensen_shannon


def test_js_identical(all_responses, response_maps):
    model, _ = all_responses
    true, _ = all_responses
    result = calculate_jensen_shannon(model, true, response_maps)
    assert result["Q1"] == 0.0
    assert result["Q2"] == 0.0


def test_js_different(all_responses, response_maps):
    model, true = all_responses
    result = calculate_jensen_shannon(model, true, response_maps)
    assert result["Q1"] == 0.0
    assert 0 < result["Q2"] < 1


def test_jensen_shannon_symmetry(all_responses, response_maps):

    model, true = all_responses
    result1 = calculate_jensen_shannon(model, true, response_maps)
    result2 = calculate_jensen_shannon(true, model, response_maps)
    assert result1["Q1"] == result2["Q1"]
    assert result1["Q2"] == result2["Q2"]


def test_wasserstein_identical(all_responses, response_maps):
    model, _ = all_responses
    true, _ = all_responses
    result = calculate_wasserstein(model, true, response_maps)
    assert result["Q1"] == 0.0
    assert result["Q2"] == 0.0


def test_wasserstein_different(all_responses, response_maps):
    model, true = all_responses
    result = calculate_wasserstein(model, true, response_maps)
    assert result["Q1"] == 0.0
    assert 0 < result["Q2"] < 1


def test_wasserstein_normalization(extreme_responses, response_maps):
    model, true = extreme_responses
    result = calculate_wasserstein(model, true, response_maps)
    assert np.isclose(result["Q1"], 1)


@pytest.fixture
def response_maps():
    return {
        "Q1": {-1: "M", 1: "A", 2: "B", 3: "C", 4: "D"},
        "Q2": {-1: "M", 1: "X", 2: "Y", 3: "Z", 4: "W"},
    }


@pytest.fixture
def all_responses():
    model = {
        "Q1": pd.Series([-1, 1, 1, 2, 2, 2, 2]),
        "Q2": pd.Series([1, 3, 3, 3, 4]),
    }
    true = {
        "Q1": pd.Series([1, 1, 2, 2, 2, 2]),  # identical to model
        "Q2": pd.Series([1, 1, 1, 4, 4]),  # different from model
    }
    return model, true


@pytest.fixture
def extreme_responses():
    # For normalization test: all mass at min vs all at max
    model = {"Q1": pd.Series([1] * 10)}
    true = {"Q1": pd.Series([4] * 10)}
    return model, true
