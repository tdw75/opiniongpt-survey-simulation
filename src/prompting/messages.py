import pandas as pd

from data.variables import get_valid_responses


def build_messages(survey_df: pd.DataFrame) -> dict[str, str]:

    prompts = {}

    for group in survey_df["group"].unique():
        question_group = survey_df[survey_df["group"] == group]
        item_stem = question_group["item_stem"].unique().item()
        responses = question_group["responses"].unique().item()
        subtopics = zip(question_group["number"], question_group["subjects"])
        prompts[group] = build_prompt_message(item_stem, responses, *subtopics)

    return prompts


def build_prompt_message(item_stem: str, response_set: dict[int, str], subtopics: list[str] | None) -> str:

    return f"""
{item_stem}

{format_subtopics(subtopics)}

{format_responses(response_set)}
"""


def format_responses(response_set: dict[int, str]) -> str:
    response_set = get_valid_responses(response_set)
    message = """The possible responses are:"""
    for key, response in response_set.items():
        message += f"\n{key}: {response}"
    return message


def format_subtopics(subtopics: list[str] | None) -> str:
    if subtopics is None:
        return "\n"
    else:
        message = """The aspects are:"""
        for s in subtopics:
            message += f"\n- {s}"
        return message