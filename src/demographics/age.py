from abc import ABC
from dataclasses import dataclass

from src.demographics.base import BaseSubGroup


@dataclass
class Age(BaseSubGroup, ABC):
    COLUMN = "Q262"


@dataclass
class Over30(Age):
    # todo: should we remove older people (45+??) or give them lower weight?
    _upper_age = 45
    VALUES = {*range(30, _upper_age + 1)}
    PERSONA = f"a person aged between 30 and {_upper_age}"


@dataclass
class OldPeople(Age):
    # old people defined as those born in or before 1980
    # note: significant overlap with Over30
    COLUMN = "Q261"  # year of birth
    VALUES = {*range(1900, 1981)}
    PERSONA = "a person aged over 45"
