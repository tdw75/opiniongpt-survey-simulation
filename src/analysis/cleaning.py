import re

import numpy as np
import pandas as pd
from sympy import separatevars

from src.data.variables import (
    split_response_string,
    ResponseMap,
    flip_key_value,
    ResponseReverseMap,
)


def clean_response():
    # todo: remove 'response', 'Your response:', question number and other common extras ()
    # todo: split into k: v - consider different formats than k: response, e.g. (k) response, k. response, etc.
    pass


def clean_generated_responses(results: pd.DataFrame) -> pd.DataFrame:
    responses = results["response"].apply(strip_response_prompt_qnum)
    responses = responses.apply(identify_unlabeled_response)
    responses = responses.apply(split_response_with_missing)

    results = separate_key_text_columns(results, responses)
    return results


def separate_key_text_columns(results: pd.DataFrame, responses: pd.Series) -> pd.DataFrame:
    results[["response_key", "response_text"]] = pd.DataFrame(responses.tolist())
    return results



def strip_response_prompt_qnum(response_string: str) -> str:
    pattern = re.compile(
        r"^\s*(your response|response|Q\d+)\s*[:\-]\s*(.+)", re.IGNORECASE
    )
    try:
        return pattern.match(response_string).group(2)
    except AttributeError:
        return response_string


def identify_unlabeled_response(response_string: str) -> str:
    pattern = re.compile("^\s*(\d+)\s*(?:[:\-\.]+)?\s*$")
    try:
        key = pattern.match(response_string).group(1)
        return f"{key}: unlabeled response"
    except AttributeError:
        return response_string


def split_response_with_missing(response):
    pattern = re.compile("^\s*(\d+)\s*[:\-\.]+\s*((?:(?!\d+\s*[:\-\.]).)+)\s*$")
    try:
        return split_response_string(response, pattern)
    except AttributeError:
        return -1, "missing"  # todo: add actual missing encoding/name


def match_outputs_with_responses(
    results: pd.DataFrame,
    responses: dict[str, ResponseMap],
    flipped_responses: dict[str, ResponseMap],
) -> pd.DataFrame:

    results["response_text"] = results["response_text"].str.strip()
    results = flip_keys_back(results, responses, flipped_responses)
    results = mark_is_correct_key_value(results, responses)
    return results


def flip_keys_back(
    results: pd.DataFrame,
    responses: dict[str, ResponseMap],
    flipped_responses: dict[str, ResponseMap],
):
    # note: needs cleaned responses otherwise will throw KeyError

    value_to_key: dict[str, ResponseReverseMap] = {}
    for q, r in responses.items():
        value_to_key[q] = flip_key_value(r)

    def _flip(qnum: str, response_key: int) -> int:
        question_responses_flipped = flipped_responses[qnum]
        response_value = question_responses_flipped[response_key]
        return value_to_key[qnum][response_value]

    reverted_keys = results.apply(
        lambda row: _flip(row["number"], row["response_key"]), axis=1
    )

    results["response_key"] = np.where(
        results["is_scale_flipped"], reverted_keys, results["response_key"]
    )

    return results


def mark_is_correct_key_value(
    results: pd.DataFrame, responses: dict[str, ResponseMap]
) -> pd.DataFrame:

    correct_text_for_key: pd.Series = results.apply(
        lambda row: responses[row["number"]][row["response_key"]], axis=1
    )
    results["is_response_valid"] = correct_text_for_key == results["response_text"]
    return results
