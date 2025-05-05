import pandas as pd
import pytest

from demographics.age import Over30, OldPeople
from demographics.sex import Female, Male
from src.demographics.country import Germany
from src.demographics.profiles import DemographicProfile


class TestDemographicProfile:

    @pytest.mark.parametrize(
        "country, age, sex, expected",
        [
            (Germany, Over30, Female, [False] * 4 + [True] + [False] * 7),
            (Germany, OldPeople, Male, [False] * 2 + [True] * 1 + [False] * 9),
            (
                None,
                Over30,
                Female,
                [False] * 4 + [True] + [False] * 5 + [True, False],
            ),
        ],
    )
    def test_filter_condition(self, mock_data, country, age, sex, expected):
        profile = DemographicProfile(country=country, age=age, sex=sex)
        mask = profile.filter_condition(mock_data)
        pd.testing.assert_frame_equal(mock_data[mask], mock_data[expected])
