import re

import numpy as np
import pandas as pd

from src.data.variables import split_response_string, remap_outputs

PLACEHOLDER_TEXT = "<no_text>"  # standardised placeholder for bare key responses


def pipeline_clean_generated_responses(results: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a sequence of cleaning functions to model-generated responses,
    extracting key-value structure and removing superfluous text/prompts.
    """
    responses = results["response"]
    responses = responses.apply(remove_prompt_prefixes)
    responses = responses.apply(detect_bare_key_without_text)
    responses = responses.apply(split_response_into_key_value)
    results = add_separate_key_and_text_columns(results, responses)
    return results


def remove_prompt_prefixes(response_string: str) -> str:
    """
    Iteratively strips leading prompts like 'Your response:', 'Q22:', etc.
    Handles multiple prefixes recursively.
    """
    # if not isinstance(response_string, str):
    #     return response_string

    pattern = re.compile(r"^\s*(your response|response|Q\d+)\s*[:\-]\s*", re.IGNORECASE)
    while True:
        match = pattern.match(response_string)
        if not match:
            break
        response_string = response_string[match.end() :]
    return response_string.strip()


def detect_bare_key_without_text(response_string: str) -> str:
    """
    Detects cases where only a key is given (e.g. '2') and fills in placeholder text.
    """

    pattern = re.compile("^\s*(\d+)\s*(?:[:\-\.]+)?\s*$")
    match = pattern.match(response_string)
    if match:
        key = match.group(1)
        return f"{key}: {PLACEHOLDER_TEXT}"
    return response_string


def split_response_into_key_value(response: str) -> tuple[int | float, str]:
    """
    Splits a response string of the form '2: Not sure' into (2, 'Not sure').
    Falls back to (np.nan, original_response) if parsing fails.
    """
    if not isinstance(response, str):
        return np.nan, response

    pattern = re.compile(r"^\s*(\d+)\s*[:\-\.]+\s*(.+?)\s*$", re.DOTALL)
    try:
        return split_response_string(response, pattern)
    except (AttributeError, ValueError):
        return np.nan, response


def add_separate_key_and_text_columns(
    results: pd.DataFrame, responses: pd.Series
) -> pd.DataFrame:
    """
    Adds response_key and response_text columns from extracted tuples.
    """
    results[["response_key", "response_text"]] = pd.DataFrame(responses.tolist())
    return results


def remap_response_keys(
    df: pd.DataFrame, key_col: str = "response_key"
) -> pd.DataFrame:
    df[key_col] = df[key_col]
    for qnum in ["Q56", "Q119"]:
        is_qnum = df["number"] == qnum
        df.loc[is_qnum, key_col] = remap_outputs(qnum, df.loc[is_qnum, key_col])
    return df
