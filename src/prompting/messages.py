from typing import re

import pandas as pd

from src.data.variables import get_valid_responses, responses_to_map
from ast import literal_eval


def extract_user_prompts_from_survey_grouped(survey_df: pd.DataFrame) -> dict[str, str]:
    """
    General prompt format for each group of questions
    """

    prompts = {}
    survey_df["group"] = survey_df["group"].combine_first(survey_df["number"])

    for group in survey_df["group"].dropna().unique():
        # todo: handle case where no subtopics only a single question; 'group' column is currently empty so it currently skips

        question_group = survey_df[survey_df["group"] == group]
        item_stem = question_group["item_stem"].unique().item()
        numbers = question_group["number"].values
        responses = literal_eval(question_group["responses"].unique().item())
        # todo: literal_eval might be inefficient here / might be redundantly repeated

        key = (
            numbers.min()
            if numbers.shape[0] == 1
            else f"{numbers.min()}-{numbers.max()}"
        )
        prompts[key] = build_user_prompt_message_grouped(
            item_stem,
            responses_to_map(responses),
            numbers,
            question_group["subtopic"].values,
        )

    return prompts


def extract_user_prompts_from_survey_individual(
    survey_df: pd.DataFrame,
) -> dict[str, str]:
    """
    General prompt format for each individual questions
    """

    prompts = {}

    for idx, question in survey_df.iterrows():
        subtopic = f"\n{question['subtopic']}" if not pd.isnull(question["group"]) else ""
        item = f"{question['item_stem']}{subtopic}"
        responses = literal_eval(question["responses"])
        prompts[question["number"]] = build_user_prompt_message_individual(
            item, responses_to_map(responses), question["number"]
        )

    return prompts


def build_user_prompt_message_grouped(
    item_stem: str,
    response_set: dict[int, str],
    numbers: list[str],
    subtopics: list[str] | None,
) -> str:
    return f"""
{item_stem}

{format_subtopics(numbers, subtopics)}

{format_responses(response_set)}

Response:
"""


def build_user_prompt_message_individual(
    item: str, response_set: dict[int, str], number: str
) -> str:
    return f"""
{number}: {item}

{format_responses(response_set)}

Response:
"""


def format_responses(response_set: dict[int, str]) -> str:
    response_set = get_valid_responses(response_set)
    message = """The possible responses are:"""
    for key, response in response_set.items():
        message += f"\n{key}: {response}"
    message += "\n\nIf you are unsure you can answer with '-1: Don't know'"
    return message


def format_subtopics(numbers: list[str], subtopics: list[str] | None) -> str:
    if subtopics is None:
        return "\n"
    else:
        message = """The aspects are:"""
        for n, s in zip(numbers, subtopics):
            message += f"\n{n}: {s}"
        return message
