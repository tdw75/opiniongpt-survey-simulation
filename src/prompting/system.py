from src.demographics.base import BaseSubGroup


def build_persona_message(subgroup: type[BaseSubGroup]) -> str:
    return f"""
    Please answer all questions as if you are {subgroup.PERSONA}.
    Make sure your responses align as closely as possible to the responses that someone from this 
    demographic is likely to give.
    """


def build_survey_context_message() -> str:

    return """
    You are taking part in a survey on personal values and will be asked a range of 
    multiple choice questions.
     
    Each question will have the following format:
    
    "
    For each of the following statements I read out, can you tell me how much you agree
    with each. Do you agree strongly, agree, disagree, or disagree strongly?
    Q33: On the whole, men make better political leaders than women do
    Q34: When a mother works for pay, the children suffer
    
    Available responses:
    1: Agree strongly
    2: Agree
    3: Disagree
    4: Strongly disagree
    "
    
    For each question, just select a single answer from the set of available responses. 
    For example:
    "Response: 
    Q33 - 4: Strongly disagree"
    Q34 - 3: Disagree"
    """


FEW_SHOT = ""  # todo: add examples to facilitate few shot learning
