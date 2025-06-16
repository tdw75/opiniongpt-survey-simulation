import pandas as pd

from src.demographics.age import Age
from src.demographics.config import ALL_COUNTRIES, ALL_AGE, ALL_SEX
from src.demographics.country import Country
from src.demographics.sex import Sex
from src.demographics.base import BaseSubGroup


def filter_by_subgroups(df: pd.DataFrame, subgroups: list[type[BaseSubGroup]]):
    """
    Selects data that belong to any of given subgroups
    """
    mask = create_filter_condition_for_subgroups(df, subgroups)
    return df.loc[mask, :].reset_index(drop=True)


def create_filter_condition_for_subgroups(df: pd.DataFrame, subgroups: list[type[BaseSubGroup]]) -> pd.Series:

    """
    Selects data that belong to any of given subgroups
    """

    conditions = [s.filter_true(df) for s in subgroups]
    mask = conditions[0]
    if len(conditions) > 1:
        for m in conditions[1:]:
            mask |= m

    return mask
