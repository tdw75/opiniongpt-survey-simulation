from abc import ABC
from dataclasses import dataclass

from src.demographics.base import BaseSubGroup


# todo: split on country of birth or country of survey?
# Q266, Q267, Q268 own country of birth, mother's, father's


@dataclass
class Country(BaseSubGroup, ABC):
    COLUMN = "B_COUNTRY_ALPHA"


@dataclass
class Germany(Country):
    VALUES = {"DEU"}


@dataclass
class UnitedStates(Country):
    VALUES = {"USA"}


@dataclass
class MiddleEast(Country):
    # todo: add all countries
    VALUES = {"CYP", "EGY", "IRN", "IRQ", "JOR", "LBN", "TUR"}
    # definition from https://en.wikipedia.org/wiki/Middle_East
    # not found: Bahrain, Israel, Kuwait, Oman, Palestine, Qatar, Saudi Arabia, Syria, UAE, Yemen


@dataclass
class LatinAmerica(Country):
    # todo: add all countries
    VALUES = {"ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GTM", "MEX", "NIC", "PER", "PRI", "URY", "VEN"}
    # definition from https://en.wikipedia.org/wiki/Latin_America
    # not found: Costa Rica, Cuba, Dominican Republic, El Salvador, Honduras, Panama, Paraguay


if __name__ == "__main__":
    print(Germany.VALUES)
