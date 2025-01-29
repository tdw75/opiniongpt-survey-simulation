import pandas as pd
import pytest

from src.data.filtering import filter_by_subgroups, filter_sequentially
from src.demographics.age import Over30
from src.demographics.country import Germany, UnitedStates
from src.demographics.sex import Male, Female


@pytest.mark.parametrize("subgroups, mask_exp", [
    ([Germany, UnitedStates], [True] * 12),
    ([Germany], [True] * 6 + [False] * 6),
    ([Germany, Female], [True] * 6 + [False] * 3 + [True] * 3),
])
def test_filter_by_subgroups(mock_data, subgroups, mask_exp):
    df_out = filter_by_subgroups(mock_data, subgroups)
    df_exp = mock_data.loc[mask_exp, :].reset_index(drop=True)
    pd.testing.assert_frame_equal(df_out, df_exp)


@pytest.mark.parametrize("countries, age, sex, mask_exp", [
    ([], [], [], [True] * 12),
    ([Germany, UnitedStates], [Over30], [Male, Female], [False, True, True] * 4),
    ([], [Over30], [], [False, True, True] * 4),
    ([Germany], [Over30], [], [False, True, True] * 2 + [False] * 6),
    ([UnitedStates], [Over30], [Female], [False] * 10 + [True] * 2),
])
def test_filter_sequentially(mock_data, countries, age, sex, mask_exp):
    df_out = filter_sequentially(mock_data, countries, age, sex)
    df_exp = mock_data.loc[mask_exp, :].reset_index(drop=True)
    pd.testing.assert_frame_equal(df_out, df_exp)
