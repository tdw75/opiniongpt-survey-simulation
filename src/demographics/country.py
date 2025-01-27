from abc import ABC
from dataclasses import dataclass

from src.demographics.base import BaseSubGroup


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
    VALUES = {"TUR"}


@dataclass
class LatinAmerica(Country):
    VALUES = {"MEX", "BRA",}


if __name__ == "__main__":
    print(Germany.VALUES)
