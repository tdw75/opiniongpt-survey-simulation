from src.demographics.sex import Female
from src.prompting.system import build_persona_message


def test_build_persona_message():
    message = build_persona_message(Female)
    expected = f"""
    Please answer all following questions as if you are a person that identifies as female.
    Make sure your responses align as closely as possible to the responses that someone from this 
    demographic is likely to give.
    """
    assert message == expected