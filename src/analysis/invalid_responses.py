import re
from enum import StrEnum

import numpy as np
import pandas as pd

from src.analysis.cleaning import remove_prompt_prefixes, PLACEHOLDER_TEXT
from src.data.variables import ResponseMap, ResponseReverseMap, flip_key_value, QNum


class InvalidReasons(StrEnum):
    MISMATCH = "key text mismatch;"
    INVALID_KEY = "invalid key;"
    INVALID_TEXT = "invalid text;"
    NO_RESPONSE = "no response given;"
    AMBIGUOUS = "ambiguous response;"


def pipeline_identify_invalid_responses(
    results: pd.DataFrame,
    responses: dict[str, ResponseMap],
    flipped_responses: dict[str, ResponseMap],
) -> pd.DataFrame:
    results["reason_invalid"] = ""

    results["response_text"] = results["response_text"].apply(normalise_response_text)
    results = flip_keys_back(results, responses, flipped_responses)
    results = extract_first_response_instance(results, responses)
    results["extra_text"] = clean_extra_text(results)
    results = recover_keys_from_text_only(results, responses)
    results = mark_multiple_responses(results, responses)
    results = mark_key_value_valid_mismatch(results, responses)
    results = mark_no_response_given(results)
    results = mark_invalid(results)
    return results


def normalise_response_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def clean_extra_text(results: pd.DataFrame) -> pd.Series:
    # run twice to catch case with both
    extra_text = results["extra_text"]
    return extra_text.apply(remove_prompt_prefixes)


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
        else:
            question_responses_flipped = flipped_responses[qnum]
            response_value = question_responses_flipped.get(response_key)
            return value_to_key[qnum].get(response_value, response_key)

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
    """
    1) Try to extract a leading valid label from response_text (primary_from_text).
    2) If no leading label AND the text is empty/placeholder, and the key is valid,
       then derive primary from the key (bare-key case).
    3) Otherwise, do not override with the key (junk/contradictory text remains invalid).
    4) extra_text gets the trailing content if extracted from text; otherwise keep original text for audit.
    """
    results = results.copy()
    primaries, extras = [], []

    for qnum, group in results.groupby("number"):
        resp_map = {k: v for k, v in valid_responses[qnum].items() if k >= 0}
        valid_keys = set(resp_map.keys())

        allowed = sorted(resp_map.values(), key=len, reverse=True)
        pat = re.compile(
            rf"^({'|'.join(map(re.escape, allowed))})(?:\s+(.*))?",
            flags=re.IGNORECASE | re.DOTALL,
        )
        extracted = group["response_text"].str.extract(pat)
        primary_from_text = extracted[0]
        extra_from_text = extracted[1]

        text_has_leading_label = primary_from_text.notna()
        text_is_placeholder = group["response_text"].eq(PLACEHOLDER_TEXT) | group[
            "response_text"
        ].eq("")
        key_is_valid = group["response_key"].isin(valid_keys)

        primary = primary_from_text.copy()
        use_key = (~text_has_leading_label) & text_is_placeholder & key_is_valid
        primary.loc[use_key] = group.loc[use_key, "response_key"].map(resp_map)

        extra = pd.Series("", index=group.index, dtype="object")
        # trailing only for rows where a label is extracted from text
        has_trail = text_has_leading_label & extra_from_text.notna()
        extra.loc[has_trail] = extra_from_text.loc[has_trail].fillna("")
        # for non-text-extracted rows, keep original text for audit
        keep_orig = (
            (~text_has_leading_label)
            & ~text_is_placeholder
            & group["response_text"].notna()
        )
        extra.loc[keep_orig] = group.loc[keep_orig, "response_text"]

        primaries.append(primary.fillna(""))
        extras.append(extra.fillna(""))

    results["response_text"] = pd.concat(primaries).sort_index()
    results["extra_text"] = pd.concat(extras).sort_index()
    return results


def extract_first_response_instance1(
    results: pd.DataFrame, valid_responses: dict[str, ResponseMap]
):
    primary_list = []
    extra_list = []
    # reason_list = []
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
        primary, extra = extracted[0], extracted[1]
        # reason = group["reason_invalid"]

        # identify missing response text
        is_text_missing = primary.isna()
        is_extra_text_missing = extra.isna()
        extra[is_text_missing] = group["response_text"][is_text_missing]
        primary[is_text_missing & is_extra_text_missing] = PLACEHOLDER_TEXT

        # add reason for invalidity
        # reason.update(reason[is_text_missing] + InvalidReasons.NO_RESPONSE)
        # reason_list.append(reason)
        primary_list.append(primary)
        extra_list.append(extra)

    results["response_text"] = pd.concat(primary_list).sort_index()
    results["extra_text"] = pd.concat(extra_list).sort_index().fillna("")
    # results["reason_invalid"] = pd.concat(reason_list).sort_index()
    return results


def mark_no_response_given(results: pd.DataFrame) -> pd.DataFrame:
    """
    Marks responses with no response text as invalid.
    """
    results = results.copy()
    is_no_response = _check_missing_text(results) & _check_missing_key(results)
    results["reason_invalid"].update(
        results["reason_invalid"][is_no_response] + InvalidReasons.NO_RESPONSE
    )

    return results


def recover_keys_from_text_only(
    results: pd.DataFrame, responses: dict[str, ResponseMap]
) -> pd.DataFrame:
    """
    Attempts to recover missing response_keys from text-only responses.
    Only applies recovery if text maps uniquely to a known label for the question.
    """
    results = results.copy()
    recovered_keys = []

    for qnum, group in results.groupby("number"):
        response_map = responses[qnum]
        text_only = group["response_key"].isna()

        recovered = group.loc[text_only, "response_text"].apply(
            lambda text: _try_recover_key_from_text(text, response_map)
        )
        recovered_keys.append(recovered)

    recovered_keys = pd.concat(recovered_keys).sort_index()
    results.loc[recovered_keys.index, "response_key"] = recovered_keys
    results["response_key"] = results["response_key"].astype("Int64")
    return results


def mark_multiple_responses(
    results: pd.DataFrame, valid_responses: dict[QNum, ResponseMap]
):
    reason_list = []
    for qnum, group in results.groupby("number"):

        responses = list(valid_responses[qnum].values())
        responses.sort(key=len, reverse=True)
        all_responses = r"(?<!\w)(" + "|".join(map(re.escape, responses)) + r")(?!\w)"

        is_multiple_responses = group["extra_text"].str.contains(
            all_responses, flags=re.IGNORECASE, regex=True
        )

        reason = group["reason_invalid"].copy()
        reason.update(reason[is_multiple_responses] + InvalidReasons.AMBIGUOUS)
        reason_list.append(reason)

    results["reason_invalid"] = pd.concat(reason_list).sort_index()
    return results


def mark_key_value_valid_mismatch(
    results: pd.DataFrame, responses: dict[str, ResponseMap]
) -> pd.DataFrame:

    results = results.copy()
    reason_list = []

    for qnum, group in results.groupby("number"):
        valid_responses = {k: v.lower() for k, v in responses[qnum].items() if k >= 0}
        reason = group["reason_invalid"].copy()
        valid_keys = set(valid_responses.keys())
        valid_texts = set(valid_responses.values())

        key_invalid = _check_invalid_key(group, valid_keys)
        text_invalid = _check_invalid_text(group, valid_texts)
        key_text_mismatch = _check_key_text_mismatch(group, valid_responses)

        reason.update(reason[key_text_mismatch] + InvalidReasons.MISMATCH)
        reason.update(reason[text_invalid] + InvalidReasons.INVALID_TEXT)
        reason.update(reason[key_invalid] + InvalidReasons.INVALID_KEY)

        reason_list.append(reason)

    results["reason_invalid"] = pd.concat(reason_list).sort_index()
    return results


def _check_invalid_key(df: pd.DataFrame, valid_keys: set[int]) -> pd.Series:
    """
    Marks responses where the response_key is missing or not among valid keys.
    """
    is_text_only = _check_missing_key(df)
    is_valid_key = df["response_key"].isin(valid_keys)
    return ~is_text_only & ~is_valid_key


def _check_missing_key(df: pd.DataFrame) -> pd.Series:
    """
    Marks responses where the response_key is missing.
    """
    return df["response_key"].isna()


def _check_missing_text(df: pd.DataFrame) -> pd.Series:
    return df["response_text"] == PLACEHOLDER_TEXT


def _check_missing_extra_text(df: pd.DataFrame) -> pd.Series:
    return df["extra_text"] == ""


def _check_invalid_text(group: pd.DataFrame, valid_texts: set[str]) -> pd.Series:
    is_placeholder = _check_missing_text(group)
    is_valid_text = group["response_text"].isin(valid_texts)
    return ~is_valid_text & ~is_placeholder


def _check_key_text_mismatch(
    group: pd.DataFrame, valid_responses: ResponseMap
) -> pd.Series:
    """
    Marks rows where a valid response_text does not match the label implied by response_key.
    Ignores rows with missing keys or incomplete (placeholder) text.
    """
    expected_text_for_key = group["response_key"].map(valid_responses)

    has_key = group["response_key"].notna()
    has_valid_text = group["response_text"].isin(valid_responses.values())
    return has_key & has_valid_text & (group["response_text"] != expected_text_for_key)


def _try_recover_key_from_text(
    response_text: str, response_map: dict[int, str]
) -> int | None:
    matches = [k for k, v in response_map.items() if v == response_text]
    return matches[0] if len(matches) == 1 else None


def mark_invalid(results: pd.DataFrame) -> pd.DataFrame:
    is_valid = results["reason_invalid"] == ""
    results.loc[is_valid, "reason_invalid"] = "valid"
    results["final_response"] = np.where(is_valid, results["response_key"], -1)
    return results
