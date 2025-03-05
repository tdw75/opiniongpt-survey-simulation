from abc import ABC
from dataclasses import dataclass

from src.demographics.base import BaseSubGroup


@dataclass
class Sex(BaseSubGroup, ABC):
    COLUMN = "Q260"


@dataclass
class Male(Sex):
    VALUES = {1}
    PERSONA = "a person that identifies as male"



@dataclass
class Female(Sex):
    VALUES = {2}
    PERSONA = "a person that identifies as female"


