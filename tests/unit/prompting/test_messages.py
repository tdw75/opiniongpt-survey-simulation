import pickle

import pandas as pd
import os

import pytest

from src.prompting.messages import build_prompt_message, format_subtopics, build_messages


def test_build_messages(expected_messages):
    # todo: handle case where no subtopics only a single question; 'group' column is currently empty so it currently skips
    survey = pd.read_csv("test_data_files/sample_variables.csv")
    indices = [*range(6)] + [*range(21, 26)]
    messages = build_messages(survey.loc[indices])
    assert messages == expected_messages



def test_build_prompt_message(expected_messages):
    survey = pd.read_csv("test_data_files/sample_variables.csv")
    question = survey.loc[0]
    subjects = survey.loc[0:5, "subtopic"].values
    numbers = survey.loc[0:5, "number"].values
    responses = {
        1: "Very important",
        2: "Rather important",
        3: "Not very important",
        4: "Not at all important",
        -1: "INVALID"
    }

    message = build_prompt_message(question["item_stem"], responses, numbers, subjects)
    assert message == expected_messages["Q1-Q6"]


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


def load_page(num: int, directory: str = "test_data_files/pages"):
    with open(os.path.join(directory, f"page{num}.pkl"), "rb") as f:
        return pickle.load(f)


@pytest.fixture
def expected_messages() -> dict[str, str]:
    q1_q6 ="""
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

If you are unsure you can answer with '-1: Don't know'
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

If you are unsure you can answer with '-1: Don't know'
"""
    q27 ="""
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

If you are unsure you can answer with '-1: Don't know'
"""
    return {"Q1-Q6": q1_q6, "Q22-Q26": q22_q26, "Q27": q27}
