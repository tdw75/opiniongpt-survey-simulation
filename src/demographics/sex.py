from abc import ABC
from dataclasses import dataclass

from src.demographics.base import BaseSubGroup


@dataclass
class Sex(BaseSubGroup, ABC):
    COLUMN = "Q260"


@dataclass
class Male(Sex):
    VALUES = {1}


@dataclass
class Female(Sex):
    VALUES = {2}


