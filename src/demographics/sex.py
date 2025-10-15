from dataclasses import dataclass

from src.demographics.base import BaseSubGroup, classproperty

column = "Q260"


@dataclass
class Male(BaseSubGroup):

    @classproperty
    def COLUMN(cls) -> str:
        return column


    @classproperty
    def ADAPTER(cls) -> str:
        return "men"

    @classproperty
    def VALUES(cls) -> set[int]:
        return {1}

    @classproperty
    def PERSONA(cls) -> str:
        return "a person that identifies as male"

@dataclass
class Female(BaseSubGroup):

    @classproperty
    def COLUMN(cls) -> str:
        return column

    @classproperty
    def ADAPTER(cls) -> str:
        return "women"

    @classproperty
    def VALUES(cls) -> set[int]:
        return {2}

    @classproperty
    def PERSONA(cls) -> str:
        return "a person that identifies as female"


ALL_SEXES = [Male, Female]