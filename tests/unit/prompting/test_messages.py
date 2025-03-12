import pickle

import pandas as pd
import os
from prompting.messages import build_prompt_message


def test_build_prompt_message():
    survey = pd.read_csv("test_data_files/variables.csv")
    question = survey.loc[0]
    subjects = survey.loc[0:5, "subject"].values
    responses = {
        1: "Very important",
        2: "Rather important",
        3: "Not very important",
        4: "Not at all important",
        -1: "INVALID"
    }

    message = build_prompt_message(question["item_stem"], responses, subjects)
    expected_message = """
For each of the following aspects, indicate how important it is in your life. Would you 
say it is very important, rather important, not very important or not important at all?

The aspects are:
- Family
- Friends
- Leisure time
- Politics
- Work
- Religion

The possible responses are:
1: Very important
2: Rather important
3: Not very important
4: Not at all important

If you are unsure you can answer with -1: Don't know
"""
    assert message == expected_message


def load_page(num: int, directory: str = "test_data_files/pages"):
    with open(os.path.join(directory, f"page{num}.pkl"), "rb") as f:
        return pickle.load(f)
