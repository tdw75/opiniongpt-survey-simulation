from abc import ABC
from dataclasses import dataclass

from src.demographics.base import BaseSubGroup, classproperty



column = ""  #todo: add column number


class Liberal(BaseSubGroup):

    @classproperty
    def COLUMN(cls) -> str:
        return column

    @classproperty
    def ADAPTER(cls) -> str:
        return "liberal"

    @classproperty
    def VALUES(cls) -> set[str]:
        return {"Left", "2", "3"}

    @classproperty
    def PERSONA(cls) -> str:
        return "a person with a progressive or politically left-wing view of the world"


class Conservative(BaseSubGroup):

    @classproperty
    def COLUMN(cls) -> str:
        return column

    @classproperty
    def ADAPTER(cls) -> str:
        return "conservative"

    @classproperty
    def VALUES(cls) -> set[str]:
        return {"8", "9", "Right"}

    @classproperty
    def PERSONA(cls) -> str:
        return "a person with a conservative or politically right-wing view of the world"