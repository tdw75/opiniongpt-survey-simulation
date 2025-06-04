from src.demographics.config import subgroups
from src.demographics.base import BaseSubGroup


def build_survey_context_message(is_with_few_shot: bool) -> str:

    message = """
    You are participating in a survey on personal values. For each question, you will be given a list of possible 
    responses after the question. Read the question and possible responses carefully. Choose the single response 
    that best fits your answer and reply using only the exact key and value pair from the provided list (including 
    number, colon, space, and text). Do not include any other words, formatting, or explanation — just the 
    selected response exactly as shown. Remember to consider your personal values and beliefs when making your selection
    """
    if is_with_few_shot:
        message += f"\n{FEW_SHOT_EXAMPLE}"

    return message


def build_persona_message(subgroup: type[BaseSubGroup]) -> str:
    return f"""
    Please answer all questions as if you are {subgroup.PERSONA}.
    Make sure your responses align as closely as possible to the responses that someone from this 
    demographic is likely to give.
    """


def build_survey_context_for_persona(subgroup: str, is_with_few_shot: bool) -> str:
    persona = build_persona_message(subgroups[subgroup]) if subgroup is not None else ""
    return build_survey_context_message(is_with_few_shot) + persona


system_first_draft = """
You are participating in a survey on personal values. For each question, you will be given a list of 
available responses after the question. Choose the single response that best fits your answer and reply using only 
the exact key and value pair from the provided list (including number, colon, space, and text). 
Do not include any other words, formatting, or explanation — just the selected response exactly as shown.
"""


FEW_SHOT_EXAMPLE = """
Q33: For each of the following statements I read out, can you tell me how much you agree
with each. Do you agree strongly, agree, disagree, or disagree strongly? — 
Carrot cake is tastier than chocolate cake

Available responses:
1: Agree strongly
2: Agree
3: Disagree
4: Strongly disagree
"""