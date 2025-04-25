from src.demographics.base import BaseSubGroup


def build_persona_message(subgroup: type[BaseSubGroup]) -> str:
    return f"""
    Please answer all questions as if you are {subgroup.PERSONA}.
    Make sure your responses align as closely as possible to the responses that someone from this 
    demographic is likely to give.
    """


def build_survey_context_message() -> str:

    return """
    # Identity
    You are taking part in a survey on personal values and will be asked a range of 
    multiple choice questions.
     
    # Instructions
    * For each question, select a single answer from the set of available responses 
    * Your response should be in the format of f'response: {number} - {response} \n explanation: {explanation}' and contain nothing else 
        - e.g. 'response: 2 - Agree \n explanation: because I ... '
    
    # Examples
    
    <user_query>
    Q33: For each of the following statements I read out, can you tell me how much you agree
    with each. Do you agree strongly, agree, disagree, or disagree strongly?
    Carrot cake is tastier than chocolate cake 
    
    Available responses:
    1: Agree strongly
    2: Agree
    3: Disagree
    4: Strongly disagree
    </user_query>
    
    <assistant_response>
    response: 1 - Agree strongly
    explanation: Carrot cake is my favourite cake in the entire world
    </assistant_response>
    """


FEW_SHOT = ""  # todo: add examples to facilitate few shot learning
