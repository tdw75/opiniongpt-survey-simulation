import numpy as np
import pandas as pd
import pytest

from src.analysis.cleaning import (
    remove_prompt_prefixes,
    split_response_into_key_value,
    detect_bare_key_without_text,
    add_separate_key_and_text_columns,
    pipeline_clean_generated_responses,
)


@pytest.mark.parametrize(
    "string, exp",
    [
        ("Your response: 1: Value", "1: Value"),
        ("Response:   9: 9", "9: 9"),
        ("Your response: 10: value value", "10: value value"),
        ("Your response: 10 ", "10"),
        ("10: 10 ", "10: 10"),
        ("Q10: Your response: 1: value", "1: value"),
        ("response:  Q10:  value", "value"),
        ("Q10: value", "value"),
    ],
)
def test_strip_response_prompt_qnum(string, exp):
    output = remove_prompt_prefixes(string)
    assert output == exp


@pytest.mark.parametrize(
    "response, expected",
    [
        ("1: value", "1: value"),
        ("1: ", "1: <no_text>"),
        (" 1 ", "1: <no_text>"),
        (" ksdf ", " ksdf "),
        ("Q1", "Q1"),
        ("Q1: ", "Q1: "),
    ],
)
def test_detect_bare_key_without_text(response, expected):
    assert detect_bare_key_without_text(response) == expected


def test_split_response_into_key_value():
    responses = pd.Series(
        [
            "1.- Very important",
            "1: Very important",
            "1:  Very important",
            "9: 9",
            "Agree",
            "1: Very important 2: Rather important",
            "2: Disagree \n\n\nQ42: Do you agree",
        ]
    )
    expected = pd.DataFrame(
        [
            (1, "Very important"),
            (1, "Very important"),
            (1, "Very important"),
            (9, "9"),
            (np.nan, "Agree"),
            (1, "Very important 2: Rather important"),
            (2, """Disagree \n\n\nQ42: Do you agree"""),
        ],
        columns=["response_key", "response_text"],
    )
    split = responses.apply(split_response_into_key_value)
    results = add_separate_key_and_text_columns(pd.DataFrame(), split)
    pd.testing.assert_frame_equal(results, expected, check_dtype=False)


def test_separate_key_text_columns():

    results = pd.DataFrame()
    responses = pd.Series([(1, "agree"), (2, "disagree"), (2, "2"), (1, "1")])
    results_out = add_separate_key_and_text_columns(results, responses)
    pd.testing.assert_series_equal(
        results_out["response_key"],
        pd.Series([1, 2, 2, 1]),
        check_names=False,
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        results_out["response_text"],
        pd.Series(["agree", "disagree", "2", "1"]),
        check_names=False,
    )


def test_clean_generated_responses():
    responses = [
        "Your response: 1: vAlue",
        "Response:   9: 9",
        "Your response: 10: Value Value",
        "10: value value",
        "Your response: 10 ",
        "10",
        "10: 10 ",
        "Q10: Your response: 1: value",
        "response:  Q10:  value",
        "Q10: value",
        "2: Disagree \n\n\nQ42: Do you agree",
        " ksdf ",
    ]
    results = pd.DataFrame({"response": responses})
    results_clean = pipeline_clean_generated_responses(results)
    expected = pd.DataFrame(
        {
            "response": responses,
            "response_key": [1, 9, 10, 10, 10, 10, 10, 1, np.nan, np.nan, 2, np.nan],
            "response_text": [
                "value",
                "9",
                "value value",
                "value value",
                "<no_text>",
                "<no_text>",
                "10",
                "value",
                "value",
                "value",
                """disagree q42: do you agree""",
                "ksdf",
            ],
        }
    )
    pd.testing.assert_frame_equal(results_clean, expected, check_dtype=False)