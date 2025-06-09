import pandas as pd
import pytest

from src.analysis.cleaning import (
    strip_response_prompt_qnum,
    split_response_with_missing,
    identify_unlabeled_response,
    match_outputs_with_responses,
    mark_is_correct_key_value,
    flip_keys_back,
    separate_key_text_columns,
)
from variables import ResponseMap


@pytest.mark.parametrize(
    "string, output",
    [
        ("Your response: 1: value", "1: value"),
        ("Response:   9: 9", "9: 9"),
        ("Your response: 10: value value", "10: value value"),
        ("Your response: 10 ", "10 "),
        ("10: 10 ", "10: 10 "),
        ("Q10: 1: value", "1: value"),
        ("Q10: value", "value"),
    ],
)
def test_strip_response(string, output):
    assert strip_response_prompt_qnum(string) == output


@pytest.mark.parametrize(
    "response, expected",
    [
        ("1: value", "1: value"),
        ("1: ", "1: unlabeled response"),
        (" 1 ", "1: unlabeled response"),
        (" ksdf ", " ksdf "),
        ("Q1", "Q1"),
        ("Q1: ", "Q1: "),
    ],
)
def test_identify_labeless_response(response, expected):
    assert identify_unlabeled_response(response) == expected


@pytest.mark.parametrize(
    "response, expected",
    [
        ("1.- Very important", (1, "Very important")),
        ("1: Very important", (1, "Very important")),
        ("1:  Very important", (1, "Very important")),
        ("9: 9", (9, "9")),
        ("1: Very important 2: Rather important", (-1, "missing")),
    ],
)
def test_split_response_string(response, expected):
    split = split_response_with_missing(response)
    assert split == expected


def test_separate_key_text_columns():

    results = pd.DataFrame()
    responses = pd.Series([(1, "agree"), (2, "disagree"), (2, "2"), (1, "1")])
    results_out = separate_key_text_columns(results, responses)
    pd.testing.assert_series_equal(
        results_out["response_key"], pd.Series([1, 2, 2, 1]), check_names=False
    )
    pd.testing.assert_series_equal(
        results_out["response_text"],
        pd.Series(["agree", "disagree", "2", "1"]),
        check_names=False,
    )


def test_flip_keys_back(mock_response_results, responses, responses_flipped):
    results_out = flip_keys_back(mock_response_results, responses, responses_flipped)

    expected = pd.Series([1, 2, 2, 1, 1] + [2, 2, 3, 3, 2], name="response_key")
    pd.testing.assert_series_equal(results_out["response_key"], expected)


def test_mark_is_correct_key_value(mock_response_results, responses):

    results_out = mark_is_correct_key_value(mock_response_results, responses)
    expected = pd.Series(
        [True, True, False, False, False] + [True, False, True, False, True],
        name="is_response_valid",
    )
    pd.testing.assert_series_equal(results_out["is_response_valid"], expected)


@pytest.fixture
def mock_response_results() -> pd.DataFrame:
    results = pd.DataFrame(
        {
            "number": ["Q1"] * 5 + ["Q2"] * 5,
            "response_key": [1, 2, 1, 1, 2] + [2, 2, 3, 1, 2],
            "response_text": ["agree", "disagree", "disagree", "potato", "agree"]
            + ["2", "potato", "3", "3", "2"],
            "is_scale_flipped": [False, False, True, False, True]
            + [False, False, False, True, True],
        }
    )
    return results


@pytest.fixture
def responses() -> dict[str, ResponseMap]:
    return {
        "Q1": {1: "agree", 2: "disagree"},
        "Q2": {1: "1", 2: "2", 3: "3"},
    }


@pytest.fixture
def responses_flipped() -> dict[str, ResponseMap]:
    return {
        "Q1": {1: "disagree", 2: "agree"},
        "Q2": {1: "3", 2: "2", 3: "1"},
    }
