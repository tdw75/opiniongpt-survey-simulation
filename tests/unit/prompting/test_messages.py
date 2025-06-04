import pickle

import pandas as pd
import os
import pytest

from src.simulation.models import ModelConfig
from src.prompting.messages import (
    build_user_prompt_message_grouped,
    format_subtopics,
    extract_user_prompts_from_survey_grouped,
    build_user_prompt_message_individual,
    extract_user_prompts_from_survey_individual,
    format_messages,
)


def test_extract_user_prompts_from_survey_grouped(expected_messages_grouped):
    # todo: handle case where no subtopics only a single question; 'group' column is currently empty so it currently skips
    survey = pd.read_csv("test_data_files/sample_variables.csv")
    indices = [*range(6)] + [*range(21, 26)]
    messages = extract_user_prompts_from_survey_grouped(survey.loc[indices], False)
    assert messages == expected_messages_grouped


@pytest.mark.parametrize(
    "file_name, is_subtopic_separate, is_reverse",
    [
        # ("sample_variables", True),
        ("sample_variables_not_split", False, True),
        ("sample_variables_not_split", False, False),
    ],
)
def test_extract_user_prompts_from_survey_individual(
    expected_messages_individual, file_name, is_subtopic_separate, is_reverse
):
    survey = pd.read_csv(f"test_data_files/{file_name}.csv")
    indices = [0, 3, 22, 25]
    messages = extract_user_prompts_from_survey_individual(
        survey.loc[indices], is_subtopic_separate, is_reverse
    )
    assert messages == expected_messages_individual[is_reverse]


def test_build_user_prompt_message_grouped(expected_messages_grouped):
    survey = pd.read_csv("test_data_files/sample_variables.csv")
    question = survey.loc[0]
    subjects = survey.loc[0:5, "subtopic"].values
    numbers = survey.loc[0:5, "number"].values
    responses = {
        1: "Very important",
        2: "Rather important",
        3: "Not very important",
        4: "Not at all important",
    }

    message = build_user_prompt_message_grouped(
        question["item_stem"], responses, numbers, subjects
    )
    assert message == expected_messages_grouped["Q1-Q6"]


@pytest.mark.parametrize("idx, number", [(0, "Q1"), (3, "Q4")])
def test_build_user_prompt_message_individual(
    expected_messages_individual, idx, number
):
    survey = pd.read_csv("test_data_files/sample_variables_not_split.csv")
    question = survey.loc[idx]
    responses = {
        1: "Very important",
        2: "Rather important",
        3: "Not very important",
        4: "Not at all important",
    }

    message = build_user_prompt_message_individual(
        f"{question['item_stem']}", responses, question["number"]
    )
    assert message == expected_messages_individual[False][number]


def test_format_subtopics():
    survey = pd.read_csv("test_data_files/sample_variables.csv")
    numbers = survey.loc[0:5, "number"].values
    subtopics = survey.loc[0:5, "subtopic"].values
    message = format_subtopics(numbers, subtopics)
    expected_message = """The aspects are:
Q1: Family
Q2: Friends
Q3: Leisure time
Q4: Politics
Q5: Work
Q6: Religion"""
    assert message == expected_message


def test_format_messages():

    system_prompt = "this is the system prompt"
    user_prompt = "what is your name?"
    formatted = format_messages(
        user_prompt, ModelConfig(base_model_name="phi", system_prompt=system_prompt)
    )
    assert formatted == [
        {"role": "user", "content": "this is the system prompt\nwhat is your name?"}
    ]


def load_page(num: int, directory: str = "test_data_files/pages"):
    with open(os.path.join(directory, f"page{num}.pkl"), "rb") as f:
        return pickle.load(f)


@pytest.fixture
def expected_messages_grouped() -> dict[str, str]:
    q1_q6 = """
For each of the following aspects, indicate how important it is in your life. Would you 
say it is very important, rather important, not very important or not important at all?

The aspects are:
Q1: Family
Q2: Friends
Q3: Leisure time
Q4: Politics
Q5: Work
Q6: Religion

The possible responses are:
1: Very important
2: Rather important
3: Not very important
4: Not at all important

Your response:
"""
    # todo: wording of neighbours question doesn't really make sense, might need to change manually
    q22_q26 = """
On this list are various groups of people. Could you please mention any that you would 
not like to have as neighbors?

The aspects are:
Q22: Homosexuals
Q23: People of a different religion
Q25: Unmarried couples living together
Q26: People who speak a different language

The possible responses are:
1: Mentioned
2: Not mentioned

Your response:
"""
    q27 = """
For each of the following statements I read out, can you tell me how much you agree 
with each. Do you agree strongly, agree, disagree, or disagree strongly? - One of my 
main goals in life has been to make my parents proud

The aspects are:
Q27: One of main goals in life has been to make my parents proud

The possible responses are:
1: Agree strongly
2: Agree
3: Disagree
4: Strongly disagree

Your response:
"""
    return {"Q1-Q6": q1_q6, "Q22-Q26": q22_q26, "Q27": q27}


@pytest.fixture
def expected_messages_individual() -> dict[bool, dict[str, str]]:
    q1 = """
Q1: For each of the following aspects, indicate how important it is in your life. Would you 
say it is very important, rather important, not very important or not important at all? –
Family

The possible responses are:
1: Very important
2: Rather important
3: Not very important
4: Not at all important

Your response:
"""
    q1_fl = """
Q1: For each of the following aspects, indicate how important it is in your life. Would you 
say it is very important, rather important, not very important or not important at all? –
Family

The possible responses are:
1: Not at all important
2: Not very important
3: Rather important
4: Very important

Your response:
"""
    q4 = """
Q4: For each of the following aspects, indicate how important it is in your life. Would you
say it is very important, rather important, not very important or not important at all? –
Politics

The possible responses are:
1: Very important
2: Rather important
3: Not very important
4: Not at all important

Your response:
"""
    q4_fl = """
Q4: For each of the following aspects, indicate how important it is in your life. Would you
say it is very important, rather important, not very important or not important at all? –
Politics

The possible responses are:
1: Not at all important
2: Not very important
3: Rather important
4: Very important

Your response:
"""
    q23 = """
Q23: On this list are various groups of people. Could you please mention any that you would
not like to have as neighbors? – People of a different religion

The possible responses are:
1: Mentioned
2: Not mentioned

Your response:
"""
    q23_fl = """
Q23: On this list are various groups of people. Could you please mention any that you would
not like to have as neighbors? – People of a different religion

The possible responses are:
1: Not mentioned
2: Mentioned

Your response:
"""
    q27 = """
Q27: For each of the following statements I read out, can you tell me how much you agree
with each. Do you agree strongly, agree, disagree, or disagree strongly? - One of my
main goals in life has been to make my parents proud

The possible responses are:
1: Agree strongly
2: Agree
3: Disagree
4: Strongly disagree

Your response:
"""
    q27_fl = """
Q27: For each of the following statements I read out, can you tell me how much you agree
with each. Do you agree strongly, agree, disagree, or disagree strongly? - One of my
main goals in life has been to make my parents proud

The possible responses are:
1: Strongly disagree
2: Disagree
3: Agree
4: Agree strongly

Your response:
"""

    return {
        False: {"Q1": q1, "Q4": q4, "Q23": q23, "Q27": q27},
        True: {"Q1": q1_fl, "Q4": q4_fl, "Q23": q23_fl, "Q27": q27_fl},
    }
