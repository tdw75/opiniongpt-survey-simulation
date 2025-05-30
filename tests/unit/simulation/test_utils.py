import pandas as pd
import pytest

from src.simulation.utils import get_run_name, filter_survey_subset


def test_filter_survey_subset():
    survey = pd.read_csv("test_data_files/sample_variables.csv")
    subsets = {"groups": ["Important in life"], "individual_questions": ["Q29", "Q30"]}
    survey_subset = filter_survey_subset(survey, subsets)
    expected_qnums = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q29", "Q30"]
    pd.testing.assert_series_equal(
        survey_subset["number"], pd.Series(expected_qnums, name="number")
    )


@pytest.mark.parametrize(
    "base_model_name, is_lora, subgroup, exp_name",
    [
        ("claude", False, "australian", "claude-instruct-australian"),
        ("claude", True, "australian", "claude-opinion-gpt-australian"),
        ("claude", False, None, "claude-instruct-general"),
    ],
)
def test_get_run_name(base_model_name, is_lora, subgroup, exp_name):
    assert get_run_name(base_model_name, is_lora, subgroup) == exp_name
