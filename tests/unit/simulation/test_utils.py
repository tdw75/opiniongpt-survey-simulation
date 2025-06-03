import pandas as pd
import pytest

from models import ModelConfig
from src.simulation.utils import filter_survey_subset, get_batch


def test_filter_survey_subset():
    survey = pd.read_csv("test_data_files/sample_variables.csv")
    subsets = {"groups": ["Important in life"], "individual_questions": ["Q29", "Q30"]}
    survey_subset = filter_survey_subset(survey, subsets)
    expected_qnums = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q29", "Q30"]
    pd.testing.assert_series_equal(
        survey_subset["number"], pd.Series(expected_qnums, name="number")
    )


@pytest.mark.parametrize(
    "sample_size, batch_size, expected",
    [(10, 3, [3, 3, 3, 1]), (10, 2, [2] * 5), (10, 10, [10]), (10, 1, [1] * 10)],
)
def test_get_batch(sample_size, batch_size, expected):
    config = ModelConfig(sample_size=sample_size, batch_size=batch_size)
    for i, batch in enumerate(get_batch(config)):
        assert batch == expected[i]
