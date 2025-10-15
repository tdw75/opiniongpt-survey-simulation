from src.demographics.age import OldPeople, Over30, ALL_AGES
from src.demographics.country import (
    Germany,
    UnitedStates,
    MiddleEast,
    LatinAmerica,
    ALL_COUNTRIES,
)
from src.demographics.sex import Female, Male, ALL_SEXES
from src.demographics.political_leaning import Liberal, Conservative, ALL_LEANINGS


subgroups = {
    "liberal": Liberal,
    "conservative": Conservative,
    "german": Germany,
    "american": UnitedStates,
    "latin_america": LatinAmerica,
    "middle_east": MiddleEast,
    "men": Male,
    "women": Female,
    "people_over_30": Over30,
    "old_people": OldPeople,
    # "teenager": "AskTeenagers",
}


dimensions = {
    "leaning": ALL_LEANINGS,
    "country": ALL_COUNTRIES,
    "sex": ALL_SEXES,
    "age": ALL_AGES,
}


def generate_q_range(lower: int, upper: int) -> list[str]:
    return [f"Q{i}" for i in range(lower, upper + 1)]


category_to_question = {
    "Social Values, Norms, Stereotypes": generate_q_range(1, 45),
    "Happiness and Wellbeing": generate_q_range(46, 56),
    "Social Capital, Trust and Organizational Membership": generate_q_range(57, 105),
    "Economic Values": generate_q_range(106, 111),
    "Perceptions of Corruption": generate_q_range(112, 120),
    "Perceptions of Migration": generate_q_range(121, 130),
    "Perceptions of Security": generate_q_range(131, 151),
    "Perceptions about Science and Technology": generate_q_range(158, 163),
    "Religious Values": generate_q_range(164, 175),
    "Ethical Values": generate_q_range(176, 198),
    "Political Interest and Political Participation": generate_q_range(199, 234),
    "Political Culture and Political Regimes": generate_q_range(235, 259),
}


categories = list(category_to_question.keys())

question_to_category: dict[str, str] = {
    qnum: category for category, qnums in category_to_question.items() for qnum in qnums
}
