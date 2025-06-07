import re

import pandas as pd

from src.data.variables import split_response_string


def clean_response():
    # todo: strip extra spaces, etc.
    # todo: remove 'response', 'Your response:', question number and other common extras ()
    # todo: split into k: v - consider different formats than k: response, e.g. (k) response, k. response, etc.
    pass


def strip_response(response_string: str) -> str:
    pattern = re.compile(
        r"^\s*(your response|response)\s*[:\-]\s*(\d+:\s*.+)", re.IGNORECASE
    )
    try:
        return pattern.match(response_string).group(2)
    except AttributeError:
        return response_string


def strip_qnum(response_string: str) -> str:
    pattern = re.compile(
        r"^\s*(your response|response)\s*[:\-]\s*(\d+:\s*.+)", re.IGNORECASE
    )
    try:
        return pattern.match(response_string).group(2)
    except AttributeError:
        return response_string


def split_response_with_missing(response):
    pattern = re.compile("^\s*(\d+)\s*[:\-\.]+\s*((?:(?!\d+\s*[:\-\.]).)+)\s*$")
    try:
        return split_response_string(response, pattern)
    except AttributeError:
        return -1, "missing"  # todo: add actual missing encoding/name


def clean_generated_responses(results: pd.DataFrame) -> pd.DataFrame:
    responses = results["response"].apply(strip_response)
    responses = responses.apply(split_response_with_missing)
    results["response_key"] = responses.apply(lambda r: r[0])
    results["response_text"] = responses.apply(lambda r: r[1])
    return results
