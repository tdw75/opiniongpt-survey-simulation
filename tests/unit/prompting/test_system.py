from src.demographics.sex import Female
from src.prompting.system import build_persona_message, build_survey_context_for_persona


def test_build_persona_message():
    message = build_persona_message(Female)
    expected = f"""
    Please answer all questions as if you are a person that identifies as female.
    Make sure your responses align as closely as possible to the responses that someone from this 
    demographic is likely to give.
    """
    assert message == expected


def test_build_survey_context_for_persona():
    message = build_survey_context_for_persona("men")
    expected = f"""
    You are participating in a survey on personal values. For each question, you will be given a list of possible 
    responses after the question. Read the question and possible responses carefully. Choose the single response 
    that best fits your answer and reply using only the exact key and value pair from the provided list (including 
    number, colon, space, and text). Do not include any other words, formatting, or explanation — just the 
    selected response exactly as shown. Remember to consider your personal values and beliefs when making your selection
    
    Please answer all questions as if you are a person that identifies as male.
    Make sure your responses align as closely as possible to the responses that someone from this 
    demographic is likely to give.
    """
    assert message == expected