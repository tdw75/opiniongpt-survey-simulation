import numpy as np
import pandas as pd
import pytest

from src.analysis.cleaning import (
    strip_leading_response_prompt_qnum,
    split_response_into_key_value,
    identify_bare_key,
    add_separate_key_and_text_columns,
)


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
