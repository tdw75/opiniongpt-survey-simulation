import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)


@dataclass
class BaseSubGroup(ABC):

    @classmethod
    def filter_true(cls, df: pd.DataFrame) -> pd.Series:
        return df[cls.COLUMN].isin(cls.VALUES)

    @classmethod
    def filter_model(cls, df: pd.DataFrame) -> pd.Series:
        return df["subgroup"] == cls.ADAPTER

    @classproperty
    @abstractmethod
    def COLUMN(cls) -> str:
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def VALUES(self) -> set[str | int]:
        raise NotImplementedError

    @classproperty
    @abstractmethod
    def PERSONA(self) -> str:
        raise NotImplementedError

    @classproperty
    def NAME(cls) -> str:
        return pascal_to_snake(cls.__name__)

    @classproperty
    @abstractmethod
    def ADAPTER(cls) -> str:
        raise NotImplementedError


def pascal_to_snake(string: str) -> str:
    return re.sub("(?!^)([A-Z]+)", r"_\1", string).lower()
