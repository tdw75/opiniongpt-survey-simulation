from abc import ABC
from dataclasses import dataclass

from src.demographics.base import BaseSubGroup, classproperty


# todo: split on country of birth or country of survey?
# Q266, Q267, Q268 own country of birth, mother's, father's


column = "B_COUNTRY_ALPHA"


@dataclass
class Germany(BaseSubGroup):

    @classproperty
    def COLUMN(cls) -> str:
        return column

    @classproperty
    def ADAPTER(cls) -> str:
        return "german"

    @classproperty
    def VALUES(cls) -> set[str]:
        return {"DEU"}

    @classproperty
    def PERSONA(cls) -> str:
        return "a person born and raised in the Federal Republic of Germany"


@dataclass
class UnitedStates(BaseSubGroup):

    @classproperty
    def COLUMN(cls) -> str:
        return column

    @classproperty
    def ADAPTER(cls) -> str:
        return "american"

    @classproperty
    def VALUES(cls) -> set[str]:
        return {"USA"}

    @classproperty
    def PERSONA(cls) -> str:
        return "a person born and raised in the United States of America"


@dataclass
class MiddleEast(BaseSubGroup):
    # definition from https://en.wikipedia.org/wiki/Middle_East
    # not found in WVS: Bahrain, Israel, Kuwait, Oman, Palestine, Qatar, Saudi Arabia, Syria, UAE, Yemen

    @classproperty
    def COLUMN(cls) -> str:
        return column

    @classproperty
    def ADAPTER(cls) -> str:
        return "middle_east"

    @classproperty
    def VALUES(cls) -> set[str]:
        return {"CYP", "EGY", "IRN", "IRQ", "JOR", "LBN", "TUR"}

    @classproperty
    def PERSONA(cls) -> str:
        return "a person born and raised in a Middle Eastern Country"


@dataclass
class LatinAmerica(BaseSubGroup):
    # definition from https://en.wikipedia.org/wiki/Latin_America
    # not found in WVS: Costa Rica, Cuba, Dominican Republic, El Salvador, Honduras, Panama, Paraguay

    @classproperty
    def COLUMN(cls) -> str:
        return column

    @classproperty
    def ADAPTER(cls) -> str:
        return "latin_american"

    @classproperty
    def VALUES(cls) -> set[str]:
        return {
            "ARG",
            "BOL",
            "BRA",
            "CHL",
            "COL",
            "ECU",
            "GTM",
            "MEX",
            "NIC",
            "PER",
            "PRI",
            "URY",
            "VEN",
        }

    @classproperty
    def PERSONA(cls) -> str:
        return "a person born and raised in a Latin American country"
