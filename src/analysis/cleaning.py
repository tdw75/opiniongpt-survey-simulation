import re

import numpy as np
import pandas as pd

from src.data.variables import (
    split_response_string,
    ResponseMap,
    flip_key_value,
    ResponseReverseMap,
)


def clean_generated_responses(results: pd.DataFrame) -> pd.DataFrame:
    responses = results["response"]
    responses = responses.apply(strip_leading_response_prompt_qnum)
    responses = responses.apply(strip_leading_response_prompt_qnum)
    # run twice to catch case with both
    responses = responses.apply(identify_keyless_response)
    responses = responses.apply(split_response_into_key_value)
    results = separate_key_and_text_columns(results, responses)
    return results


def separate_key_and_text_columns(
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


def identify_keyless_response(response_string: str) -> str:
    pattern = re.compile("^\s*(\d+)\s*(?:[:\-\.]+)?\s*$")
    try:
        key = pattern.match(response_string).group(1)
        return f"{key}: keyless response"
    except AttributeError:
        return response_string


def split_response_into_key_value(response):
    # pattern = re.compile("^\s*(\d+)\s*[:\-\.]+\s*((?:(?!\d+\s*[:\-\.]).)+)\s*$")
    pattern = re.compile(r"^\s*(\d+)\s*[:\-\.]+\s*(.*)$", re.DOTALL)
    try:
        return split_response_string(response, pattern)
    except AttributeError:  # assign key -1 if no key found
        return -1, response


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
    results = strip_extra_continuations(results, responses)
    results = mark_is_correct_key_value(results, responses)
    # todo: add one more step including responses like 4: some explanation -- check that explanation is not another response
    # todo: check that keyless responses are considered and not automatically deemed invalid
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


def strip_extra_continuations(
    results: pd.DataFrame, valid_responses: dict[str, ResponseMap]
):
    cleaned_list = []
    reason_list = []
    for qnum, group in results.groupby("number"):

        # regex
        allowed = valid_responses[qnum].values()
        valid_pattern = "|".join(map(re.escape, allowed))
        main_pattern = rf"^({valid_pattern})\s+(Q\d+:|(response|your response):).*"
        invalid_pattern = rf"^({valid_pattern})\s+\d+:\s+({valid_pattern}).*"

        # extract valid responses
        extracted = group["response_text"].str.extract(
            main_pattern, flags=re.IGNORECASE | re.DOTALL
        )
        cleaned = extracted[0]
        reason = group["reason_invalid"]

        # identify invalid responses
        is_na = cleaned.isna()
        is_multiple_responses = group["response_text"].str.match(
            invalid_pattern, case=False, na=False, flags=re.DOTALL
        )

        # fill cleaned column
        cleaned[is_multiple_responses] = group["response_text"][is_multiple_responses]
        cleaned[is_na] = group["response_text"][is_na]
        cleaned_list.append(cleaned)

        # add reason for invalidity
        reason.loc[is_multiple_responses] += "multiple responses;"
        reason_list.append(reason)
        print(is_multiple_responses.sum())

    results["response_text"] = pd.concat(cleaned_list).sort_index()
    results["reason_invalid"] = pd.concat(reason_list).sort_index()
    return results


def mark_is_correct_key_value(
    results: pd.DataFrame, responses: dict[str, ResponseMap]
) -> pd.DataFrame:

    def _get_correct_key_value(row: pd.Series) -> str:
        return responses[row["number"]].get(row["response_key"], -1)

    correct_text_for_key: pd.Series = results.apply(_get_correct_key_value, axis=1)
    is_correct_text_for_key = correct_text_for_key == results["response_text"]
    results["is_response_valid"] = np.where(
        results["response_key"] >= 0, is_correct_text_for_key, False
    )
    results.loc[~is_correct_text_for_key, "reason_invalid"] += "wrong key;"
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
    is_invalid = ~results["is_response_valid"]
    results["response_key"] = np.where(is_invalid, -1, results["response_key"])
    return results


def add_missing_response_keys(
    responses: dict[str, ResponseMap],
) -> dict[str, ResponseMap]:
    for qnum, _ in responses.items():
        responses[qnum][-1] = "missing"
    return responses
