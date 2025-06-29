import numpy as np
import pandas as pd
import pytest

from src.analysis.invalid_responses import (
    flip_keys_back,
    extract_first_response_instance,
    mark_multiple_responses,
    mark_is_correct_key_value,
    mark_is_correct_key_value,
)
from src.data.variables import ResponseMap


def test_flip_keys_back(mock_response_results, responses, responses_flipped):
    results_out = flip_keys_back(mock_response_results, responses, responses_flipped)

    expected = pd.Series(
        [1, 2, 2, pd.NA, 1, 1, pd.NA, 2, 3, 3, -1, -1],
        name="response_key"
    ).astype("Int64")

    pd.testing.assert_series_equal(results_out["response_key"], expected, check_dtype=False)


def test_extract_first_response_instance():
    reason = "no response given;"
    numbers = ["Q1"] * 7 + ["Q2"] * 6
    n = len(numbers)
    valid = {
        "Q1": {1: "agree", 2: "agree strongly", 3: "disagree"},
        "Q2": {1: "1", 2: "2", 3: "3"},
    }
    results = pd.DataFrame(
        {
            "number": numbers,
            "reason_invalid": [""] * n,
            "response_text": [
                "agree \n\n\n  Q2: what",
                "agree\n\n\n  3: disagree",
                "agree strongly",
                "no idea",
                "disagree  sorry I cannot",
                "2: Disagree\n\n\nQ42: Do you agree",
                "agree Q2: response: hello",
            ]
            + [
                "3  Q3: hi",
                "3    Response: hello",
                "disagree  I am an AI",
                "2  3: 3",
                "1   your response: hello",
                "1   your response: 2",
            ],
        }
    )
    expected = pd.DataFrame(
        {
            "number": numbers,
            "reason_invalid": ["", "", "", reason, "", reason, ""]
            + ["", "", reason, "", "", ""],
            "response_text": [
                "agree",
                "agree",
                "agree strongly",
                "",
                "disagree",
                "",
                "agree",
            ]
            + ["3", "3", "", "2", "1", "1"],
            "extra_text": [
                "Q2: what",
                "3: disagree",
                "",
                "no idea",
                "sorry I cannot",
                "2: Disagree\n\n\nQ42: Do you agree",
                "Q2: response: hello",
            ]
            + [
                "Q3: hi",
                "Response: hello",
                "disagree  I am an AI",
                "3: 3",
                "your response: hello",
                "your response: 2",
            ],
        }
    )
    out = extract_first_response_instance(results, valid)
    print(out[out.columns[1:]])
    pd.testing.assert_frame_equal(out, expected)


def test_mark_multiple_responses():
    reason = "multiple responses;"
    numbers = ["Q1"] * 6 + ["Q2"] * 6
    n = len(numbers)
    valid = {
        "Q1": {1: "agree", 2: "disagree", 3: "strongly disagree"},
        "Q2": {1: "1", 2: "2", 3: "3"},
    }
    response_text = [
        "Q2: what",
        "I strongly agree with you",  # should match
        "3: disagree",
        "sorry I cannot",
        "I strongly agreeably",
        "Q42: Do you agree",
    ] + [
        "Q3: hi",
        "Q3: 3",
        "disagree  I am an AI",
        "3: 3",
        "your response: hello",
        "your response: 2",
    ]
    results = pd.DataFrame(
        {"number": numbers, "reason_invalid": [""] * n, "extra_text": response_text}
    )
    expected = pd.DataFrame(
        {
            "number": numbers,
            "reason_invalid": ["", reason, reason, "", "", reason]
            + ["", reason, "", reason, "", reason],
            "extra_text": response_text,
        }
    )
    out = mark_multiple_responses(results, valid)
    print(out)
    pd.testing.assert_frame_equal(out, expected)



def test_mark_is_correct_key_value(mock_response_results, responses):
    mock_response_results["reason_invalid"] = ""
    reason1 = "key text mismatch;"
    reason2 = "invalid response;"
    reason3 = "invalid key;"

    # last response hardcoded as -1, missing
    results_out = mark_is_correct_key_value(mock_response_results, responses)
    expected_reason = pd.Series(
        ["", "", reason1, "", "", reason2]
        + [reason2, reason2, "", reason1, reason1+reason3, reason1+reason3],
        name="reason_invalid",
    )
    print(results_out[["response_key", "response_text", "reason_invalid"]])
    pd.testing.assert_series_equal(results_out["reason_invalid"], expected_reason)


@pytest.fixture
def mock_response_results() -> pd.DataFrame:
    results = pd.DataFrame(
        {
            "number": ["Q1"] * 6 + ["Q2"] * 6,
            "response_key": [1, 2, 1, pd.NA, 2, 1] + [pd.NA, 2, 3, 1, 5, -1],
            "response_text": [
                "agree",
                "disagree",
                "disagree",
                "agree",
                "key without response",
                "",
            ]
            + ["", "potato", "3", "3", "2", "3"],
            "is_scale_flipped": [False, False, True, False, True, False]
            + [False, False, False, True, True, True],
        }
    )
    return results


@pytest.fixture
def responses() -> dict[str, ResponseMap]:
    return {
        "Q1": {1: "agree", 2: "disagree", -1: "missing"},
        "Q2": {1: "1", 2: "2", 3: "3", -1: "missing"},
    }


@pytest.fixture
def responses_flipped() -> dict[str, ResponseMap]:
    return {
        "Q1": {1: "disagree", 2: "agree", -1: "missing"},
        "Q2": {1: "3", 2: "2", 3: "1", -1: "missing"},
    }
