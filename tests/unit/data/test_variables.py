import pickle

import pandas as pd
import pytest

from src.data.variables import strip_title, split_on_questions, pipeline, split_question_into_parts, \
    identify_question_group, Question


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


def load_page(num: int):
    with open(f"test_data_files/pages/page{num}.pkl", "rb") as f:
        return pickle.load(f)


class TestSplitQuestionsIntoParts:

    pages = {i: load_page(i) for i in range(10, 13)}
    questions = pipeline(pages)
    invalid_responses = [
        "-1-.- Don´t know",
        "-2-.- No answer",
        "-4-.- Not asked in this country",
        "-5-.- Missing; Not available"
    ]

    @pytest.mark.parametrize("num, subject", [
        (1, "Family"),
        (3, "Leisure time"),
        (4, "Politics"),
    ])
    def test_important_in_life(self, num, subject):
        question = split_question_into_parts(self.questions[num - 1])
        expected_responses = [
            "1.- Very important", "2.- Rather important", "3.- Not very important", "4.- Not at all important"
        ]
        question = Question(**question)

        assert question.number == f"Q{num}"
        assert question.name == f"Important in life: {subject}"
        assert question.group == "Important in life"
        assert question.prompt.startswith("For each of the following aspects,")
        assert question.prompt.endswith(subject)
        assert question.responses == expected_responses + self.invalid_responses

    @pytest.mark.parametrize("num, subject", [
        (11, "Imagination"),
        (13, "Thrift saving money and things"),
    ])
    def test_important_child_qualities(self, num, subject):
        question = split_question_into_parts(self.questions[num - 1])
        expected_responses = ["2.- Not mentioned", "1.- Important"]
        question = Question(**question)

        assert question.number == f"Q{num}"
        assert question.name == f"Important child qualities: {subject}"
        assert question.group == "Important child qualities"
        assert question.prompt.startswith("Here is a list of qualities ")
        assert question.prompt.endswith(subject)
        assert question.responses == expected_responses + self.invalid_responses


@pytest.mark.parametrize("name, group_exp", [
    ("Important in life: blah blah blah", "Important in life"),
    ("blah blah blah", ""),
])
def test_identify_question_group(name, group_exp):
    group, sub = identify_question_group(name)
    assert group == group_exp
    assert sub == "blah blah blah"


def test_integration():
    pages = {i: load_page(i) for i in range(10, 13)}
    questions = pipeline(pages)

    assert questions[0].startswith("Q1 Important in life: Family")
    assert questions[2].startswith("Q3 Important in life: Leisure time")
    assert questions[11].startswith("Q12 Important child qualities: Tolerance and respect for other people")
    assert pd.Series(questions).str.match("Q\d+").all()
