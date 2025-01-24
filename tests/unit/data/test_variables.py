import pickle

import pandas as pd
import pytest

from data.variables import strip_title, split_on_questions, pipeline, split_question_into_parts, identify_question_group


def test_strip_title():
    page10 = load_page(10)
    page_without_title = strip_title(page10)
    assert page_without_title.startswith("  \n \nCore Variable s")


def test_split_on_questions():

    page10 = load_page(10)
    stripped_page = strip_title(page10)
    split_questions = split_on_questions(stripped_page)
    assert split_questions[0].startswith("Core Variable s")
    assert split_questions[1].startswith("Q1 Important in life: Family")
    assert split_questions[2].startswith("Q2 Important in life: Friends")
    assert split_questions[3].startswith("Q3 Important in life: Leisure time")
    assert split_questions[4].startswith("Q4 Important in life: Politics")


@pytest.mark.parametrize("num, subject", [(1, "Family"), (3, "Leisure time"), (4, "Politics")])
def test_split_question_into_parts(num, subject):
    questions = pipeline(load_page(10))
    question = split_question_into_parts(questions[num - 1])
    expected_responses = [
        "1.- Very important",
        "2.- Rather important",
        "3.- Not very important",
        "4.- Not at all important",
        "-1-.- Don´t know",
        "-2-.- No answer",
        "-4-.- Not asked in this country",
        "-5-.- Missing; Not available"
    ]

    assert question.number == f"Q{num}"
    assert question.name == f"Important in life: {subject}"
    assert question.group == f"Important in life: {subject}"
    assert question.prompt.startswith("For each of the following aspects,")
    assert question.prompt.endswith(subject)
    assert question.responses == expected_responses


@pytest.mark.parametrize("name, group_exp", [
    ("Important in life: blah blah blah", "Important in life"),
    ("blah blah blah", "blah blah blah"),

])
def test_identify_question_group(name, group_exp):
    group, sub = identify_question_group(name)
    assert group == group_exp
    assert sub == "blah blah blah"


def test_integration():
    pages = [load_page(i) for i in range(10, 13)]
    questions = pipeline(pages)
    for idx, q in enumerate(questions):
        print("="*20, idx)
        print(q)

    assert questions[0].startswith("Q1 Important in life: Family")
    assert questions[2].startswith("Q3 Important in life: Leisure time")
    assert questions[11].startswith("Q12 Important child qualities: Tolerance and respect for other people")
    assert pd.Series(questions).str.match("Q\d+").all()


def load_page(num: int):
    with open(f"test_data_files/pages/page{num}.pkl", "rb") as f:
        return pickle.load(f)
