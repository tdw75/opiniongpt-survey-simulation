from abc import ABC
from dataclasses import dataclass

from src.demographics.base import BaseSubGroup, classproperty




@dataclass
class Over30(BaseSubGroup):
    # todo: should we remove older people (45+??) or give them lower weight?
    _upper_age = 45
    VALUES = {*range(30, _upper_age + 1)}

    @classproperty
    def COLUMN(cls) -> str:
        return "Q262"

    @classproperty
    def VALUES(cls) -> set[int]:
        return {*range(30, cls._upper_age + 1)}

    @classproperty
    def PERSONA(cls) -> str:
        return f"a person aged between 30 and {cls._upper_age}"

    @classproperty
    def ADAPTER(cls) -> str:
        return "people_over_30"


@dataclass
class OldPeople(BaseSubGroup):
    # old people defined as those born in or before 1980
    # note: significant overlap with Over30

    @classproperty
    def COLUMN(cls) -> str:
        return "Q261"

    @classproperty
    def VALUES(cls) -> set[int]:
        return {*range(1900, 1981)}

    @classproperty
    def PERSONA(cls) -> str:
        return "a person aged over 45"

    @classproperty
    def ADAPTER(cls) -> str:
        return "old_people"
