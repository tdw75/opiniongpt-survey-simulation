from ast import literal_eval

import pandas as pd

from src.data.variables import responses_to_map
from src.simulation.models import ModelConfig


Messages = list[dict[str, str]]


def extract_user_prompts_from_survey_grouped(
    survey_df: pd.DataFrame, is_reverse: bool
) -> dict[str, str]:
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
            responses_to_map(responses, is_reverse),
            numbers,
            question_group["subtopic"].values,
        )

    return prompts


def extract_user_prompts_from_survey_individual(
    survey_df: pd.DataFrame, is_subtopic_separate: bool, is_reverse: bool
) -> dict[str, str]:
    """
    General prompt format for each individual questions
    """

    prompts = {}

    for idx, question in survey_df.iterrows():
        subtopic = (
            f"\n{question['subtopic']}" if not pd.isnull(question["group"]) else ""
        )
        item = f"{question['item_stem']}{subtopic if is_subtopic_separate else ''}"
        responses = literal_eval(question["responses"])
        prompts[question["number"]] = build_user_prompt_message_individual(
            item, responses_to_map(responses, is_reverse), question["number"]
        )
        print(f"successfully loaded {question['number']}")

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

Your response:
"""


def build_user_prompt_message_individual(
    item: str, response_set: dict[int, str], number: str
) -> str:
    return f"""
{number}: {item}

{format_responses(response_set)}

Your response:
"""


def format_responses(response_set: dict[int, str]) -> str:
    message = """The possible responses are:"""
    for key, response in response_set.items():
        message += f"\n{key}: {response}"
    return message


def format_subtopics(numbers: list[str], subtopics: list[str] | None) -> str:
    if subtopics is None:
        return "\n"
    else:
        message = """The aspects are:"""
        for n, s in zip(numbers, subtopics):
            message += f"\n{n}: {s}"
        return message


def batch_messages(user_prompts: list[str], config: ModelConfig) -> list[Messages]:
    """
    takes a list of user prompts and returns a list of alternative Messages
    e.g. [user1, user2] -> [message1, message2, message1, message2, message1, message2]
    """

    messages = []
    for prompt in user_prompts:
        messages.append(format_messages(prompt, config))

    # note: if config.sample_size is odd then actual number of outputs will be config.sample_size - 1
    return messages * (config.sample_size // len(user_prompts))


def format_messages(user_prompt: str, config: ModelConfig) -> Messages:
    if config.is_phi_model:
        return [{"role": "user", "content": f"{config.system_prompt}\n{user_prompt}"}]
    else:
        return [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
