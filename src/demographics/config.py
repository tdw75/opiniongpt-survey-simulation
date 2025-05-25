from src.demographics.age import OldPeople, Over30
from src.demographics.country import Germany, UnitedStates, MiddleEast, LatinAmerica
from src.demographics.sex import Female, Male
from src.demographics.political_leaning import Liberal, Conservative

ALL_COUNTRIES = [Germany, UnitedStates, MiddleEast, LatinAmerica]
ALL_SEX = [Male, Female]
ALL_AGE = [Over30, OldPeople]
ALL_POLITICAL = [Liberal, Conservative]


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
