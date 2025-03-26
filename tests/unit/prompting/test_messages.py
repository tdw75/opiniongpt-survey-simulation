import pickle

import pandas as pd
import os

import pytest

from src.prompting.messages import build_prompt_message, format_subtopics


def test_build_prompt_message(expected_messages):
    survey = pd.read_csv("test_data_files/sample_variables.csv")
    question = survey.loc[0]
    subjects = survey.loc[0:5, "subject"].values
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
    subtopics = survey.loc[0:5, "subject"].values
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
    return {"Q1-Q6": q1_q6}
