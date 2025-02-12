import re
from dataclasses import dataclass


@dataclass
class Question:
    number: str
    name: str
    group: str
    prompt: str
    responses: list[str]


@dataclass
class QuestionPatterns:
    number = re.compile("(Q\d+)")  # todo: account for other variable groups e.g. Gxx
    name = re.compile("([A-Z].*)\\n")
    group = re.compile("(.*):\s")
    prompt = re.compile("([\s\S]*?)(?=\d\.-)")
    responses = re.compile("(-?\d+\\.-.*|-?\d+-\\.-.*)")


@dataclass
class HeaderPatterns:
    main = re.compile("\s+\n \n\d+ \n \nThe WORLD VALUES SURVEY ASSOCIATION\s+\nwww.worldvaluessurvey.org")
    sub = re.compile(r"([A-Z].*\s+\(Q\d+ *-Q\d+\))")


def strip_header(page: str, pattern) -> str:
    """
    Removes the page heading/subheading from the input string
    """
    stripped_page = re.sub(pattern, '', page)
    return stripped_page


def concatenate_pages(pages: list[str]) -> str:
    return ''.join(pages)


def split_on_questions(page: str) -> list[str]:

    """
    Splits a string containing whole page(s) of the questionnaire into a list of strings
    where each string contains a separate question and/or subheading
    """

    question_pattern = re.compile(r"(?=\nQ\d+\s+[A-Z])")
    split_page = re.split(question_pattern, page)
    return [s.strip("\n ") for s in split_page]


def filter_out_non_questions(strings: list[str]) -> list[str]:
    """
    Removes non-question strings (such as subheadings, etc.) from the input list
    """
    question_start_pattern = re.compile(r"Q\d+ *[A-Z]")
    questions = [s for s in strings if re.match(question_start_pattern, s)]
    return questions


def pipeline(pages: dict[int, str]) -> list[str]:
    # todo: add question splitting to pipeline, then update unit test
    stripped_pages = [strip_header(page, HeaderPatterns.main) for page in pages.values()]
    stripped_pages = [strip_header(page, HeaderPatterns.sub) for page in stripped_pages]
    all_pages = concatenate_pages(stripped_pages)
    questions = split_on_questions(all_pages)
    questions = filter_out_non_questions(questions)
    return questions


def split_question_into_parts(question: str) -> dict:
    number, rest = [a.strip() for a in re.split(QuestionPatterns.number, question) if a]
    name, rest = [b.strip() for b in re.split(QuestionPatterns.name, rest, maxsplit=1) if b]
    group, _ = identify_question_group(name)
    prompt, rest = [c.strip() for c in re.split(QuestionPatterns.prompt, rest, maxsplit=1) if c]
    responses = [d.strip() for d in re.split(QuestionPatterns.responses, rest)]
    return dict(
        number=number,
        name=name,
        group=group,
        prompt=prompt,
        responses=[r for r in responses if r]
    )


def identify_question_group(question_name: str) -> tuple[str, str]:
    splits: list[str] = re.split(QuestionPatterns.group, question_name, maxsplit=1)
    if len(splits) == 1:
        group, subquestion = "", splits[0]
    else:
        group, subquestion = splits[1:]
    return group, subquestion


def responses_to_map(responses: list[str]) -> dict[int, str]:
    pattern = re.compile("(-?\d+).+?([A-Z].*)")
    response_tuples = [re.split(pattern, r)[1:-1] for r in responses]
    return {int(k): v for k, v in response_tuples}


def get_invalid_responses(response_map: dict[int, str]) -> dict[int, str]:
    """
    Invalid responses are encoded with negative integers. Returns all such responses from the inpout dictionary
    """
    return {k: v for k, v in response_map.items() if k < 0}


def get_valid_responses(response_map: dict[int, str]) -> dict[int, str]:
    """
    Valid responses are encoded with non-negative integers. Returns all such responses from the inpout dictionary
    """
    return {k: v for k, v in response_map.items() if k >= 0}
