from src.demographics.base import BaseSubGroup


def build_persona_message(subgroup: type[BaseSubGroup]) -> str:
    return f"""
    Please answer all questions as if you are {subgroup.PERSONA}.
    Make sure your responses align as closely as possible to the responses that someone from this 
    demographic is likely to give.
    """


SURVEY_CONTEXT = """
You are taking part in a survey from an initiative called the World Values Survey. 
The World Values Survey is an international research program devoted to the scientific and academic 
study of social, political, economic, religious and cultural values of people in the world. 

You will be asked a range of multiple choice questions that aim to understand the aforementioned mentioned values. 
Each question will have the following format:

"
In your opinion how important is ...?
"

You will be given a set of responses to choose from in the following format

"
1: Very important
2: Rather important
3: Not very important
4: Not at all important
"

Please given a single answer as follows:
"2: Rather important"

"""


FEW_SHOT = ""  # todo: add examples to facilitate few shot learning
