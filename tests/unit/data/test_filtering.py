import pandas as pd
import pytest

from src.data.filtering import filter_by_subgroups
from src.demographics.country import Germany, UnitedStates
from src.demographics.sex import Female


@pytest.mark.parametrize("subgroups, mask_exp", [
    ([Germany, UnitedStates], [True] * 12),
    ([Germany], [True] * 6 + [False] * 6),
    ([Germany, Female], [True] * 6 + [False] * 3 + [True] * 3),
])
def test_filter_by_subgroups(mock_data, subgroups, mask_exp):
    df_out = filter_by_subgroups(mock_data, subgroups)
    df_exp = mock_data.loc[mask_exp, :].reset_index(drop=True)
    pd.testing.assert_frame_equal(df_out, df_exp)
