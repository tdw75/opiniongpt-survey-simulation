import re

import numpy as np
import pandas as pd

from src.data.variables import split_response_string


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
    except AttributeError:  # assign nan key if no key found
        return np.nan, response


def mark_missing(results: pd.DataFrame) -> pd.DataFrame:
    # todo: not used anywhere
    name_map = {"response_key": "raw_key", "response_text": "raw_text"}
    results.rename(columns=name_map)
    is_valid = results["is_response_valid"]
    results["response_key"] = np.where(is_valid, results["raw_key"], -1)
    results["response_text"] = np.where(is_valid, results["raw_text"], "missing")
    return results


