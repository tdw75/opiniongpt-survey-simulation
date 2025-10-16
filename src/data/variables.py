import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

ResponseMap = dict[int, str]
ResponseReverseMap = dict[str, int]
ResponseTuple = tuple[int, str]
QNum = str


@dataclass
class Question:
    number: QNum
    name: str
    group: str
    subtopic: str
    item_stem: str
    responses: list[str]


@dataclass
class QuestionPatterns:
    number = re.compile("(Q\d+)")  # todo: account for other variable groups e.g. Gxx
    name = re.compile("([A-Z].*)\\n")
    group = re.compile("(.*):\s")
    item_stem = re.compile("([\s\S]*?)(?=\d\.-)")
    responses = re.compile("(-?\d+\\.-.*|-?\d+-\\.-.*)")


@dataclass
class HeaderPatterns:
    main = re.compile(
        "\s+\n \n\d+ \n \nThe WORLD VALUES SURVEY ASSOCIATION\s+\nwww.worldvaluessurvey.org"
    )
    sub = re.compile(r"([A-Z].*\s+\(Q\d+ *-Q\d+\))")


def strip_header(page: str, pattern) -> str:
    """
    Removes the page heading/subheading from the input string
    """
    stripped_page = re.sub(pattern, "", page)
    return stripped_page


def concatenate_pages(pages: list[str]) -> str:
    return "".join(pages)


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
    stripped_pages = [
        strip_header(page, HeaderPatterns.main) for page in pages.values()
    ]
    stripped_pages = [strip_header(page, HeaderPatterns.sub) for page in stripped_pages]
    all_pages = concatenate_pages(stripped_pages)
    questions = split_on_questions(all_pages)
    questions = filter_out_non_questions(questions)
    return questions


def split_question_into_parts(question: str) -> dict:
    number, rest = [a.strip() for a in re.split(QuestionPatterns.number, question) if a]
    name, rest = [
        b.strip() for b in re.split(QuestionPatterns.name, rest, maxsplit=1) if b
    ]
    group, subtopic = identify_question_group(name)
    item_stem, rest = [
        c.strip() for c in re.split(QuestionPatterns.item_stem, rest, maxsplit=1) if c
    ]
    # item_stem = item_stem.replace(subtopic, "").rstrip(" \n-–:")
    responses = [d.strip() for d in re.split(QuestionPatterns.responses, rest)]
    return dict(
        number=number,
        name=name,
        group=group,
        subtopic=subtopic,
        item_stem=item_stem,
        responses=[r for r in responses if r],
    )


def identify_question_group(question_name: str) -> tuple[str, str]:
    splits: list[str] = re.split(QuestionPatterns.group, question_name, maxsplit=1)
    if len(splits) == 1:
        group, subquestion = "", splits[0]
    else:
        group, subquestion = splits[1:]
    return group, subquestion


def split_response_string(response: str, pattern) -> ResponseTuple:
    # todo: handle case "1: Very important 2: Rather important"
    match = pattern.match(response)
    return int(match.group(1)), match.group(2)


def responses_to_map(
    responses: list[str], is_scale_flipped: bool, is_only_valid: bool = True
) -> ResponseMap:
    response_pattern = re.compile("(-?\d+)(?:[^\w]*\s+)(.*)")
    response_tuples: list[ResponseTuple] = [
        split_response_string(r, response_pattern) for r in responses
    ]
    if is_only_valid:
        response_tuples = get_valid_responses(response_tuples)

    sorted_responses = sorted(response_tuples)
    values = [v for _, v in sorted_responses]
    keys = [k for k, _ in sorted_responses]
    if is_scale_flipped:
        values = values[::-1]
    return dict(zip(keys, values))


def flip_key_value(mapping: dict[Any, Any]) -> dict[Any, Any]:
    return {v: k for k, v in mapping.items()}


def get_invalid_responses(response_map: ResponseMap) -> dict[int, str]:
    """
    Invalid responses are encoded with negative integers. Returns all such responses from the inpout dictionary
    """
    return {k: v for k, v in response_map.items() if k < 0}


def get_valid_responses(
    response_tuples: list[ResponseTuple],
) -> list[ResponseTuple]:
    """
    Valid responses are encoded with non-negative integers. Returns all such responses from the inpout response tuples
    """

    return [r for r in response_tuples if r[0] >= 0]


def non_ordinal_qnums() -> list[str]:
    not_ordinal = [
        *range(7, 27),
        57,
        *range(91, 94),
        149,
        150,
        *range(152, 158),
        173,
        174,
        175,
        223,
    ]
    binary = [139, 140, 141, 144, 145, 151, 165, 166, 167, 168]
    partially_ordinal = [221, 222, 254]
    # ordinal but not equidistant - 94-105, 171-172
    return [f"Q{i}" for i in not_ordinal + binary + partially_ordinal]


def ordinal_qnums() -> list[str]:
    all_qnums = [f"Q{i}" for i in range(1, 260)]
    return [qnum for qnum in all_qnums if qnum not in non_ordinal_qnums()]


def remap_response_maps(responses: dict[QNum, ResponseMap]) -> dict[QNum, ResponseMap]:
    """
    Remaps the response maps for specific questions according to predefined remappings.
    """
    return {qnum: _remap_response_map(qnum, resp) for qnum, resp in responses.items()}


def _remap_response_map(qnum: QNum, responses: ResponseMap) -> ResponseMap:
    response_remappings = _response_remappings().get(qnum, {})
    return {response_remappings.get(k, k): v for k, v in responses.items()}


def remap_outputs(qnum: QNum, outputs: pd.Series) -> pd.Series:
    response_remappings = _response_remappings().get(qnum, {})
    return outputs.replace(response_remappings)


def _response_remappings() -> dict[QNum, dict]:
    return {"Q56": {3: 2, 2: 3}, "Q119": {0: 3, 3: 4, 4: 5}}
