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
    Q33: In your opinion how important is ...?
    
    Available responses:
    1: Very important
    2: Rather important
    3: Not very important
    4: Not at all important
    "
    
    For each question, just select a single answer from the set of available responses. 
    For example:
    "Response: Q33 - 2: Rather important"
    """


FEW_SHOT = ""  # todo: add examples to facilitate few shot learning
