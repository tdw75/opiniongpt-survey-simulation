from abc import ABC
from dataclasses import dataclass

from demographics.base import BaseSubGroup


@dataclass
class PoliticalLeaning(BaseSubGroup, ABC):
    COLUMN = ""


class Progressive(PoliticalLeaning):
    VALUES = {"Left", "2", "3"}
    PERSONA = "a person with a progressive or politically left-wing view of the word"


class Conservative(PoliticalLeaning):
    VALUES = {"8", "9", "Right"}
    PERSONA = "a person with a conservative or politically right-wing view of the word"
