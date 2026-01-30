import numpy as np
import pandas as pd
import pytest

from src.analysis.cleaning import PLACEHOLDER_TEXT
from src.analysis.invalid_responses import (
    flip_keys_back,
    extract_first_response_instance,
    mark_multiple_responses,
    mark_key_value_valid_mismatch,
    recover_keys_from_text_only,
    pipeline_identify_invalid_responses,
)
from src.data.variables import ResponseMap


def test_flip_keys_back(mock_response_results, responses, responses_flipped):
    results_out = flip_keys_back(mock_response_results, responses, responses_flipped)

    expected = pd.Series(
        [1, 2, 2, pd.NA, 1, 1, pd.NA, 2, 3, 3, 5, -1], name="response_key"
    ).astype("Int64")

    pd.testing.assert_series_equal(
        results_out["response_key"], expected, check_dtype=False
    )


def test_extract_first_response_instance():
    numbers = ["Q1"] * 7 + ["Q2"] * 6
    valid = {
        "Q1": {1: "agree", 2: "agree strongly", 3: "disagree"},
        "Q2": {1: "1", 2: "2", 3: "3"},
    }
    results = pd.DataFrame(
        {
            "number": numbers,
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
            "response_key": [1, pd.NA, 1, pd.NA, 3, 3, 1] + [3, 3, pd.NA, pd.NA, 1, pd.NA],
        }
    )
    expected = pd.DataFrame(
        {
            "number": numbers,
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
            "response_key": [1, pd.NA, 1, pd.NA, 3, 3, 1] + [3, 3, pd.NA, pd.NA, 1, pd.NA],
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
    reason = "ambiguous response;"
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


def test_mark_is_key_value_valid_match(mock_response_results, responses):
    mock_response_results["reason_invalid"] = ""
    reason1 = "key text mismatch;"
    reason2 = "invalid text;"
    reason3 = "invalid key;"

    # last response hardcoded as -1, missing
    results_out = mark_key_value_valid_mismatch(mock_response_results, responses)
    expected_reason = pd.Series(
        ["", "", reason1, "", "", reason2]
        + [reason2, reason2, "", reason1, reason1 + reason3, reason1 + reason3],
        name="reason_invalid",
    )
    results_out["reason_expected"] = expected_reason
    pd.testing.assert_series_equal(results_out["reason_invalid"], expected_reason)


def test_recover_keys_from_text_only():
    responses = {"Q1": {1: "agree", 2: "disagree"}, "Q2": {1: "1", 2: "2", 3: "3"}}
    df = pd.DataFrame(
        {
            "number": ["Q1", "Q1", "Q1", "Q2", "Q2", "Q2", "Q2"],
            "response_key": [np.nan, np.nan, 1, np.nan, np.nan, np.nan, 2],
            "response_text": ["agree", "invalid", "agree", "2", "3", "maybe", "2"],
        }
    )

    expected_keys = pd.Series(
        [1, np.nan, 1, 2, 3, np.nan, 2], dtype="Int64", name="response_key"
    )
    output_df = recover_keys_from_text_only(df, responses)
    pd.testing.assert_series_equal(output_df["response_key"], expected_keys)


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
                "<no_text>",
                "",
            ]
            + ["", "potato", "3", "3", "2", "3"],
            "is_scale_flipped": [False, False, True, False, True, False]
            + [False, False, False, True, True, True],
        }
    )
    return results


def test_pipeline_comprehensive_invalid_reasons(mock_df, responses, responses_flipped):
    responses["Q3"] = {1: "a really long response text that may get truncated"}
    responses_flipped["Q3"] = {1: "a really long response text that may get truncated"}

    out = pipeline_identify_invalid_responses(
        mock_df.copy(), responses, responses_flipped
    )
    expected_keys = pd.Series(
        [-1, 2, -1, -1, -1, 1, 2, -1, 1, -1] + [2, -1, -1, 3, 2] + [1, -1],
        name="final_response",
    )
    expected_reasons = pd.Series(
        [
            "key text mismatch;",
            "valid",
            "key text mismatch;invalid key;",
            "invalid text;",
            "ambiguous response;",
            "valid",
            "valid",
            "key text mismatch;",
            "valid",
            "ambiguous response;",
        ]
        + [
            "valid",
            "ambiguous response;",
            "key text mismatch;invalid key;",
            "valid",
            "valid",
        ]
        + ["valid", "invalid text;"],
        name="reason_invalid",
    )
    pd.testing.assert_series_equal(out["final_response"], expected_keys)
    pd.testing.assert_series_equal(out["reason_invalid"], expected_reasons)


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


@pytest.fixture
def mock_df() -> pd.DataFrame:
    c = ["number", "response_key", "response_text", "is_scale_flipped"]
    data = [
        # --- Q1 valid & invalid pairs ---
        {c[0]: "Q1", c[1]: 2, c[2]: "agree", c[3]: False},  # -1
        {c[0]: "Q1", c[1]: 2, c[2]: "disagree", c[3]: False},  # 2
        {c[0]: "Q1", c[1]: 3, c[2]: "agree", c[3]: False},  # -1
        {c[0]: "Q1", c[1]: 1, c[2]: "nonsense", c[3]: False},  # -1
        {c[0]: "Q1", c[1]: 1, c[2]: "agree  2: disagree", c[3]: False},  # -1
        {c[0]: "Q1", c[1]: np.nan, c[2]: "agree", c[3]: False},  # 1
        {c[0]: "Q1", c[1]: 1, c[2]: "disagree", c[3]: True},  # 2
        {c[0]: "Q1", c[1]: 1, c[2]: "agree", c[3]: True},  # -1
        {c[0]: "Q1", c[1]: 1, c[2]: "agree because reasons", c[3]: False},  # 1
        {c[0]: "Q1", c[1]: 1, c[2]: "agree and I also agree", c[3]: False},  # -1
        # --- Q2 valid & invalid pairs ---
        {c[0]: "Q2", c[1]: 2, c[2]: "2", c[3]: False},  # 2
        {c[0]: "Q2", c[1]: 2, c[2]: "2  then also 3", c[3]: False},  # -1
        {c[0]: "Q2", c[1]: 4, c[2]: "3", c[3]: True},  # -1
        {c[0]: "Q2", c[1]: 1, c[2]: "3", c[3]: True},  # 1
        {c[0]: "Q1", c[1]: 2, c[2]: PLACEHOLDER_TEXT, c[3]: False},  # 2
        # --- Q3 valid & invalid pairs ---
        {
            c[0]: "Q3",
            c[1]: 1,
            c[2]: "a really long response text that may ge",
            c[3]: False,
        },  # 1
        {
            c[0]: "Q3",
            c[1]: 1,
            c[2]: "a really long response text that could ge",
            c[3]: True,
        },  # -1
    ]
    return pd.DataFrame(data, dtype="object").astype({"response_key": "Int64"})
