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
