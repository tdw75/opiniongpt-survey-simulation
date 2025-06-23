import re
from enum import StrEnum

import numpy as np
import pandas as pd

from src.analysis.cleaning import strip_leading_response_prompt_qnum
from src.data.variables import ResponseMap, ResponseReverseMap, flip_key_value


class InvalidReasons(StrEnum):
    MISMATCH = "key text mismatch;"
    INVALID_KEY = "invalid key;"
    INVALID_RESPONSE = "invalid response;"
    NO_RESPONSE = "no response given;"
    MULTIPLE = "multiple responses;"


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
    # todo: edit distance higher than length for each response if no exact match
    # todo: just split on \n and evaluate the first section - if no key-value match -> invalid
    results = mark_invalid(results)
    return results


def flip_keys_back(
    results: pd.DataFrame,
    responses: dict[str, ResponseMap],
    flipped_responses: dict[str, ResponseMap],
):
    # note: expects cleaned data; will raise KeyError if mappings are missing
    results = results.copy()
    value_to_key: dict[str, ResponseReverseMap] = {}
    for q, r in responses.items():
        value_to_key[q] = flip_key_value(r)

    def _flip(qnum: str, response_key: int) -> int | type(pd.NA):
        if pd.isna(response_key):
            return pd.NA
        elif response_key < 0:
            return -1
        question_responses_flipped = flipped_responses[qnum]
        response_value = question_responses_flipped.get(response_key, "missing")
        if response_value == "missing":
            print(qnum)
            print(response_key)
            print(question_responses_flipped)
        return value_to_key[qnum][response_value]

    original_keys = results["response_key"]
    reverted_keys = results.apply(
        lambda row: _flip(row["number"], row["response_key"]), axis=1
    )

    results["response_key"] = np.where(
        results["is_scale_flipped"], reverted_keys, original_keys
    )
    results["response_key"] = results["response_key"].astype("Int64")

    return results


def extract_first_response_instance(
    results: pd.DataFrame, valid_responses: dict[str, ResponseMap]
):
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

    results = results.copy()
    reason_list = []

    for qnum, group in results.groupby("number"):
        valid_responses = {k: v for k, v in responses[qnum].items() if k >= 0}
        reason = group["reason_invalid"].copy()
        correct_text_for_key = results["response_key"].map(valid_responses).fillna("")

        is_valid_text = group["response_text"].isin(valid_responses.values())
        is_valid_key = group["response_key"].isin(valid_responses.keys())
        is_key_text_match = results["response_text"] == correct_text_for_key
        is_key_without_text = group["response_text"] == "key without response"
        is_text_without_key = group["response_key"].isna()

        is_invalid_key_text_mismatch = (
            ~(is_key_text_match | is_text_without_key) & is_valid_text
        )
        is_invalid_wrong_text = ~is_valid_text & ~is_key_without_text
        is_invalid_wrong_key = ~is_valid_key & ~is_text_without_key

        reason.loc[is_invalid_key_text_mismatch] += InvalidReasons.MISMATCH
        reason.loc[is_invalid_wrong_text] += InvalidReasons.INVALID_RESPONSE
        reason.loc[is_invalid_wrong_key] += InvalidReasons.INVALID_KEY
        reason_list.append(reason)

    results["reason_invalid"] = pd.concat(reason_list).sort_index()
    return results


def mark_invalid(results: pd.DataFrame) -> pd.DataFrame:
    is_valid = results["reason_invalid"] == ""
    results["response_value"] = np.where(is_valid, results["response_key"], -1)
    return results


def add_missing_response_keys(
    responses: dict[str, ResponseMap],
) -> dict[str, ResponseMap]:
    for qnum, _ in responses.items():
        responses[qnum][-1] = "missing"
    return responses
