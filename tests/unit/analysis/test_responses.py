import pandas as pd
import pytest

from src.analysis.responses import (
    get_true_responses_for_subgroup,
    get_model_responses_for_subgroup,
    get_response_distribution,
)
from src.demographics.country import Germany
from variables import ResponseMap


def test_get_true_responses_for_subgroup(mock_true_results, expected_true_responses):
    responses = get_true_responses_for_subgroup(
        mock_true_results, Germany, qnums=["Q20", "Q21"]
    )
    assert responses.keys() == expected_true_responses.keys()
    for key in responses.keys():
        pd.testing.assert_series_equal(
            responses[key], expected_true_responses[key], check_index=False
        )


def test_get_model_responses_for_subgroup(mock_model_results):
    responses = get_model_responses_for_subgroup(
        mock_model_results, Germany, qnums=["Q20"]
    )
    expected_responses = {
        "Q20": pd.Series([3, 3, 3, -1, 2, 2, 1, 3, 3, -1], name="response_key"),
    }
    assert responses.keys() == expected_responses.keys()
    for key in responses.keys():
        pd.testing.assert_series_equal(
            responses[key], expected_responses[key], check_index=False
        )


@pytest.mark.parametrize(
    "is_normalize, is_include_invalid, expected",
    [
        (
            False,
            True,
            {
                "Q20": {-1: 1, 1: 1, 2: 2, 3: 4, 4: 0},
                "Q21": {-1: 3, 1: 1, 2: 2, 3: 2, 4: 0, 5: 0},
            },
        ),
        (
            False,
            False,
            {
                "Q20": {1: 1, 2: 2, 3: 4, 4: 0},
                "Q21": {1: 1, 2: 2, 3: 2, 4: 0, 5: 0},
            },
        ),
        (
            True,
            True,
            {
                "Q20": {-1: 0.125, 1: 0.125, 2: 0.25, 3: 0.5, 4: 0.0},
                "Q21": {-1: 0.375, 1: 0.125, 2: 0.25, 3: 0.25, 4: 0.0, 5: 0.0},
            },
        ),
    ],
)
def test_get_response_distribution(
    expected_true_responses, response_maps, is_normalize, is_include_invalid, expected
):
    response_dists = get_response_distribution(
        expected_true_responses, response_maps, is_normalize, is_include_invalid
    )
    assert response_dists == expected


@pytest.fixture
def mock_true_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "B_COUNTRY_ALPHA": ["USA"] * 4 + ["DEU"] * 8,
            "Q20": [1, 3, 2, 3] + [1, 2, 3, 3, 3, 3, -1, 2],
            "Q21": [-1, 3, 3, 4] + [-1, 2, 2, 1, 3, 3, -1, -1],
        }
    )


@pytest.fixture
def mock_model_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "number": ["Q21"] * 2 + ["Q20"] * 10,
            "subgroup": ["american"] * 2 + ["german"] * 10,
            "response_key": [1, 4] + [3, 3, 3, -1, 2, 2, 1, 3, 3, -1],
        }
    )


@pytest.fixture
def expected_true_responses() -> dict:
    return {
        "Q20": pd.Series([1, 2, 3, 3, 3, 3, -1, 2], name="Q20"),
        "Q21": pd.Series([-1, 2, 2, 1, 3, 3, -1, -1], name="Q21"),
    }


@pytest.fixture
def response_maps() -> dict[str, ResponseMap]:
    return {
        "Q20": {-1: "Missing", 1: "1", 2: "2", 3: "3", 4: "4"},
        "Q21": {-1: "Missing", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e"},
    }
