from src.demographics.config import subgroups
from src.demographics.base import BaseSubGroup


def build_survey_context_message() -> str:

    return """
    You are taking part in a survey on personal values and will be asked a range of 
    multiple choice questions.
     
    Each question will have the following format:
    
    "
    Q33: For each of the following statements I read out, can you tell me how much you agree
    with each. Do you agree strongly, agree, disagree, or disagree strongly?
    Carrot cake is tastier than chocolate cake
    
    Available responses:
    1: Agree strongly
    2: Agree
    3: Disagree
    4: Strongly disagree
    "
    
    For each question, just select a single answer from the set of available responses.
    Please copy the answer exactly as written in the 'Available responses' list above. 
    Do not change spelling or capitalisation. 
    For example:
    4: Agree strongly"
    """


def build_persona_message(subgroup: type[BaseSubGroup]) -> str:
    return f"""
    Please answer all questions as if you are {subgroup.PERSONA}.
    Make sure your responses align as closely as possible to the responses that someone from this 
    demographic is likely to give.
    """


def build_survey_context_for_persona(subgroup: str) -> str:
    persona = build_persona_message(subgroups[subgroup])
    return build_survey_context_message()  + persona


FEW_SHOT = ""  # todo: add examples to facilitate few shot learning
