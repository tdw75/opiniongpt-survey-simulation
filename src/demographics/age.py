from abc import ABC
from dataclasses import dataclass

from src.demographics.base import BaseSubGroup


@dataclass
class Age(BaseSubGroup, ABC):
    COLUMN = "Q262"


@dataclass
class Over30(Age):
    VALUES = {*range(30, 120)}


@dataclass
class OldPeople(Age):
    # old people defined as those born in or before 1980
    # note: significant overlap with Over30
    COLUMN = "Q261"  # year of birth
    VALUES = {*range(1900, 1981)}
