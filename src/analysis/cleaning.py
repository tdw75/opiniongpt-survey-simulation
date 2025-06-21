import re
from enum import Enum, StrEnum

import numpy as np
import pandas as pd

from src.data.variables import (
    split_response_string,
    ResponseMap,
    flip_key_value,
    ResponseReverseMap,
)


class InvalidReasons(StrEnum):
    WRONG_KEY = "wrong key;"
    NO_RESPONSE = "no response given;"
    MULTIPLE = "multiple responses;"


def clean_generated_responses(results: pd.DataFrame) -> pd.DataFrame:
    responses = results["response"]
    responses = responses.apply(strip_leading_response_prompt_qnum)
    responses = responses.apply(strip_leading_response_prompt_qnum)
    # run twice to catch case with both
    responses = responses.apply(identify_bare_key)
    responses = responses.apply(split_response_into_key_value)
    results = add_separate_key_and_text_columns(results, responses)
    return results


def add_separate_key_and_text_columns(
    results: pd.DataFrame, responses: pd.Series
) -> pd.DataFrame:
    results[["response_key", "response_text"]] = pd.DataFrame(responses.tolist())
    return results


def strip_leading_response_prompt_qnum(response_string: str) -> str:
    pattern = re.compile(
        r"^\s*(your response|response|Q\d+)\s*[:\-]\s*(.+)", re.IGNORECASE
    )
    try:
        return pattern.match(response_string).group(2)
    except AttributeError:
        return response_string


def identify_bare_key(response_string: str) -> str:
    pattern = re.compile("^\s*(\d+)\s*(?:[:\-\.]+)?\s*$")
    try:
        key = pattern.match(response_string).group(1)
        return f"{key}: key without response"
    except AttributeError:
        return response_string


def split_response_into_key_value(response):
    # pattern = re.compile("^\s*(\d+)\s*[:\-\.]+\s*((?:(?!\d+\s*[:\-\.]).)+)\s*$")
    pattern = re.compile(r"^\s*(\d+)\s*[:\-\.]+\s*(.*)$", re.DOTALL)
    try:
        return split_response_string(response, pattern)
    except AttributeError:  # assign key -1 if no key found
        return -1, response


def clean_extra_text(results: pd.DataFrame) -> pd.Series:
    # run twice to catch case with both
    extra_text = results["extra_text"]
    extra_text = extra_text.apply(strip_leading_response_prompt_qnum)
    return extra_text.apply(strip_leading_response_prompt_qnum)


def match_outputs_with_valid_responses(
    results: pd.DataFrame,
    responses: dict[str, ResponseMap],
    flipped_responses: dict[str, ResponseMap],
) -> pd.DataFrame:
    results["reason_invalid"] = ""

    responses = add_missing_response_keys(responses)
    flipped_responses = add_missing_response_keys(flipped_responses)

    results["response_text"] = results["response_text"].str.strip()
    results = flip_keys_back(results, responses, flipped_responses)
    results = extract_first_response_instance(results, responses)
    results["extra_text"] = clean_extra_text(results)
    results = mark_multiple_responses(results, responses)
    results = mark_is_correct_key_value(results, responses)
    # todo: check that keyless responses are considered and not automatically deemed invalid
    # todo: edit distance higher than length for each response if no exact match
    # todo: just split on \n and evaluate the first section - if no key-value match -> invalid
    results = mark_invalid(results)
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
        if response_key < 0:
            return -1
        question_responses_flipped = flipped_responses[qnum]
        response_value = question_responses_flipped.get(response_key, "missing")
        return value_to_key[qnum][response_value]

    original_keys = results["response_key"]
    reverted_keys = results.apply(
        lambda row: _flip(row["number"], row["response_key"]), axis=1
    )

    results["response_key"] = np.where(
        results["is_scale_flipped"], reverted_keys, original_keys
    )

    return results


def extract_first_response_instance(
    results: pd.DataFrame, valid_responses: dict[str, ResponseMap]
):
    # todo: check that keys without response text are not deemed invalid
    cleaned_list = []
    extra_list = []
    reason_list = []
    for qnum, group in results.groupby("number"):

        # case: llm starts generating the next question
        allowed_responses = list(valid_responses[qnum].values())
        allowed_responses.sort(key=len, reverse=True)
        allowed = "|".join(map(re.escape, allowed_responses))
        main_pattern = rf"^({allowed})(?:\s+(.*))?"

        # extract valid responses
        extracted = group["response_text"].str.extract(
            main_pattern, flags=re.IGNORECASE | re.DOTALL
        )
        cleaned = extracted[0]
        extra = extracted[1]
        reason = group["reason_invalid"]

        # identify missing responses
        is_no_response = cleaned.isna()
        cleaned[is_no_response] = ""
        extra[is_no_response] = group["response_text"][is_no_response]

        # add reason for invalidity
        reason.loc[is_no_response] += InvalidReasons.NO_RESPONSE
        reason_list.append(reason)
        cleaned_list.append(cleaned)
        extra_list.append(extra)

    results["response_text"] = pd.concat(cleaned_list).sort_index()
    results["extra_text"] = pd.concat(extra_list).sort_index().fillna("")
    results["reason_invalid"] = pd.concat(reason_list).sort_index()
    return results


def mark_multiple_responses(
    results: pd.DataFrame, valid_responses: dict[str, ResponseMap]
):
    reason_list = []
    for qnum, group in results.groupby("number"):

        responses = list(valid_responses[qnum].values())
        responses.sort(key=len, reverse=True)
        all_responses = r"(?<!\w)(" + "|".join(map(re.escape, responses)) + r")(?!\w)"

        is_multiple_responses = group["extra_text"].str.contains(
            all_responses, flags=re.IGNORECASE, regex=True
        )
        reason = group["reason_invalid"]
        # add reason for invalidity
        reason.loc[is_multiple_responses] += InvalidReasons.MULTIPLE
        reason_list.append(reason)

    results["reason_invalid"] = pd.concat(reason_list).sort_index()
    return results


def mark_is_correct_key_value(
    results: pd.DataFrame, responses: dict[str, ResponseMap]
) -> pd.DataFrame:

    def _get_correct_key_value(row: pd.Series) -> str:
        return responses[row["number"]].get(row["response_key"], "")

    correct_text_for_key: pd.Series = results.apply(_get_correct_key_value, axis=1)
    is_correct_text_for_key = results["response_text"] == correct_text_for_key
    is_bare_key = results["response_text"] == "key without response"
    results.loc[
        ~(is_correct_text_for_key | is_bare_key), "reason_invalid"
    ] += InvalidReasons.WRONG_KEY
    # todo: make sure 'missing' responses are marked properly
    return results


def mark_missing(results: pd.DataFrame) -> pd.DataFrame:
    name_map = {"response_key": "raw_key", "response_text": "raw_text"}
    results.rename(columns=name_map)
    is_valid = results["is_response_valid"]
    results["response_key"] = np.where(is_valid, results["raw_key"], -1)
    results["response_text"] = np.where(is_valid, results["raw_text"], "missing")
    return results


def mark_invalid(results: pd.DataFrame) -> pd.DataFrame:
    is_valid = results["reason_invalid"]==""
    results["response_value"] = np.where(is_valid, results["response_key"], -1)
    return results


def add_missing_response_keys(
    responses: dict[str, ResponseMap],
) -> dict[str, ResponseMap]:
    for qnum, _ in responses.items():
        responses[qnum][-1] = "missing"
    return responses
