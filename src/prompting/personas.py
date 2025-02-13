from demographics.base import BaseSubGroup


def build_persona_message(subgroup: type[BaseSubGroup]) -> str:
    return f"""
    Please answer all following questions as if you are {subgroup.PERSONA}.
    Make sure your responses align as closely as possible to the responses that someone from this 
    demographic is likely to give.
    """