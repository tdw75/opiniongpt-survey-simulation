import pandas as pd
import pytest

from src.analysis.cleaning import (
    strip_leading_response_prompt_qnum,
    split_response_into_key_value,
    identify_bare_key,
    mark_is_correct_key_value,
    flip_keys_back,
    add_separate_key_and_text_columns,
    extract_first_response_instance,
    InvalidReasons,
    mark_multiple_responses,
)
from variables import ResponseMap


@pytest.mark.parametrize(
    "string, exp1, exp2",
    [
        ("Your response: 1: value", "1: value", None),
        ("Response:   9: 9", "9: 9", None),
        ("Your response: 10: value value", "10: value value", None),
        ("Your response: 10 ", "10 ", None),
        ("10: 10 ", "10: 10 ", None),
        ("Q10: Your response: 1: value", "Your response: 1: value", "1: value"),
        ("response:  Q10:  value", "Q10:  value", "value"),
        ("Q10: value", "value", None),
    ],
)
def test_strip_response_prompt_qnum(string, exp1, exp2):
    once = strip_leading_response_prompt_qnum(string)
    twice = strip_leading_response_prompt_qnum(once)
    exp2 = exp2 or exp1
    assert once == exp1
    assert twice == exp2


@pytest.mark.parametrize(
    "response, expected",
    [
        ("1: value", "1: value"),
        ("1: ", "1: key without response"),
        (" 1 ", "1: key without response"),
        (" ksdf ", " ksdf "),
        ("Q1", "Q1"),
        ("Q1: ", "Q1: "),
    ],
)
def test_identify_bare_key(response, expected):
    assert identify_bare_key(response) == expected


@pytest.mark.parametrize(
    "response, expected",
    [
        ("1.- Very important", (1, "Very important")),
        ("1: Very important", (1, "Very important")),
        ("1:  Very important", (1, "Very important")),
        ("9: 9", (9, "9")),
        (
            "1: Very important 2: Rather important",
            (1, "Very important 2: Rather important"),
        ),
        (
            """2: Disagree


Q42: Do you agree""",
            (
                2,
                """Disagree


Q42: Do you agree""",
            ),
        ),
    ],
)
def test_split_response_into_key_value(response, expected):
    split = split_response_into_key_value(response)
    assert split == expected


def test_separate_key_text_columns():

    results = pd.DataFrame()
    responses = pd.Series([(1, "agree"), (2, "disagree"), (2, "2"), (1, "1")])
    results_out = add_separate_key_and_text_columns(results, responses)
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

    expected = pd.Series([1, 2, 2, 1, 1, 1] + [2, 2, 3, 3, 2, -1], name="response_key")
    pd.testing.assert_series_equal(results_out["response_key"], expected)


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
    reason = InvalidReasons.WRONG_KEY

    # last response hardcoded as -1, missing
    results_out = mark_is_correct_key_value(mock_response_results, responses)
    expected_valid = pd.Series(
        [True, True, False, False, False, False]
        + [True, False, True, False, True, False],
        name="is_response_valid",
    )
    expected_reason = pd.Series(
        ["", "", reason, reason, reason, reason] + ["", reason, "", reason, "", ""],
        name="reason_invalid",
    )
    pd.testing.assert_series_equal(results_out["is_response_valid"], expected_valid)
    pd.testing.assert_series_equal(results_out["reason_invalid"], expected_reason)


@pytest.fixture
def mock_response_results() -> pd.DataFrame:
    results = pd.DataFrame(
        {
            "number": ["Q1"] * 6 + ["Q2"] * 6,
            "response_key": [1, 2, 1, 1, 2, 1] + [2, 2, 3, 1, 2, -1],
            "response_text": [
                "agree",
                "disagree",
                "disagree",
                "agree I am an ai",
                "agree",
                "",
            ]
            + ["2", "potato", "3", "3", "2", "missing"],
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
