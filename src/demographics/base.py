import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class BaseSubGroup(ABC):

    @classmethod
    def filter_condition(cls, df: pd.DataFrame) -> pd.Series:
        return df[cls.COLUMN].isin(cls.VALUES)

    @property
    @abstractmethod
    def COLUMN(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def VALUES(self) -> set[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def PERSONA(self) -> set[str]:
        raise NotImplementedError

    @classmethod
    @property
    def NAME(cls):
        return pascal_to_snake(cls.__name__)


def pascal_to_snake(string: str) -> str:
    return re.sub("(?!^)([A-Z]+)", r"_\1", string).lower()
