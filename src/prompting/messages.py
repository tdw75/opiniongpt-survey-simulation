import pandas as pd

from src.data.variables import get_valid_responses, responses_to_map
from ast import literal_eval


def build_messages(survey_df: pd.DataFrame) -> dict[str, str]:
    """
    General prompt format for each group of questions
    """
    # todo: add unit test

    prompts = {}

    for group in survey_df["group"].dropna().unique():
        question_group = survey_df[survey_df["group"] == group]
        item_stem = question_group["item_stem"].unique().item()

        # todo: literal_eval might be inefficient here / might be redundantly repeated
        responses = literal_eval(question_group["responses"].unique().item())
        # todo: handle case where no subtopics, just question again
        prompts[group] = build_prompt_message(
            item_stem, responses_to_map(responses), question_group["number"].values, question_group["subtopic"].values
        )

    return prompts


def build_prompt_message(item_stem: str, response_set: dict[int, str], numbers: list[str], subtopics: list[str] | None) -> str:

    return f"""
{item_stem}

{format_subtopics(numbers, subtopics)}

{format_responses(response_set)}
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
