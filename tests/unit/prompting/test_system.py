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
You are taking part in a survey on personal values and will be asked a range of multiple choice questions.  

Each question will have the following format:

"
Q33: For each of the following statements I read out, can you tell me how much you agree
with each. Do you agree strongly, agree, disagree, or disagree strongly? — 
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
4: Agree strongly


Please answer all questions as if you are a person that identifies as male.
Make sure your responses align as closely as possible to the responses that someone from this 
demographic is likely to give.
"""
    assert message == expected