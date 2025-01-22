import re
from dataclasses import dataclass


@dataclass
class Question:
    number: str
    name: str
    prompt: str
    responses: list[str]


@dataclass
class QuestionPatterns:
    number: str = re.compile("(Q\d+)")
    name: str = re.compile("([A-Z].*)\\n")
    prompt: str = re.compile("([\s\S]*)(?=1\\.-)")  #fixme: hardcoded 1 works
    # prompt: str = re.compile("([\s\S]*)(?=\d\\.-)")  #fixme: \d doesn't for some reason, only finds from 4.-
    responses: str = re.compile("(-?\d+\\.-.*|-?\d+-\\.-.*)")
    # fixme: not filtering our '' for some reason


def strip_title(page: str) -> str:
    """
    Removes the page header from the input string and then strips leading whitespace/new lines from the output
    """
    header_pattern = re.compile("\s+\n \n\d+ \n \nThe WORLD VALUES SURVEY ASSOCIATION\s+\nwww.worldvaluessurvey.org")
    stripped_page = re.sub(header_pattern, '', page)
    return stripped_page


def concatenate_pages(pages: list[str]) -> str:
    return ''.join(pages)


def split_on_questions(page: str) -> list[str]:

    """
    Splits a string containing whole page(s) of the questionnaire into a list of strings
    where each string contains a separate question and/or subheading
    """

    question_pattern = re.compile(r"(?=\nQ\d+ [A-Z])")
    split_page = re.split(question_pattern, page)
    return [s.strip("\n ") for s in  split_page]
    # return split_page


def filter_out_non_questions(strings: list[str]) -> list[str]:
    """
    Removes non-question strings (such as subheadings, etc.) from the input list
    """
    question_start_pattern = re.compile(r"Q\d+ [A-Z]")
    questions = [s for s in strings if re.match(question_start_pattern, s)]
    return questions


def pipeline(pages: list[str]) -> list[str]:
    # todo: add question splitting to pipeline, then update unit test
    stripped_pages = [strip_title(page) for page in pages]
    all_pages = concatenate_pages(stripped_pages)
    questions = split_on_questions(all_pages)
    questions = filter_out_non_questions(questions)
    return questions


def split_question_into_parts(question: str) -> Question:
    number, rest = [a.strip() for a in re.split(QuestionPatterns.number, question) if a]
    name, rest = [b.strip() for b in re.split(QuestionPatterns.name, rest, maxsplit=1) if b]
    prompt, rest = [c.strip() for c in re.split(QuestionPatterns.prompt, rest, maxsplit=1) if c]
    responses = [d.strip() for d in re.split(QuestionPatterns.responses, rest)]
    return Question(number=number, name=name, prompt=prompt, responses=[r for r in responses if r])
