from abc import ABC
from dataclasses import dataclass

from demographics.base import BaseSubGroup


@dataclass
class PoliticalLeaning(BaseSubGroup, ABC):
    COLUMN = ""


class Progressive(PoliticalLeaning):
    VALUES = {"Left", "2", "3"}


class Conservative(PoliticalLeaning):
    VALUES = {"8", "9", "Right"}
