import pandas as pd

from src.simulation.utils import filter_survey_subset, mark_is_scale_flipped


def test_filter_survey_subset():
    survey = pd.read_csv("test_data_files/sample_variables.csv")
    subsets = {"groups": ["Important in life"], "individual_questions": ["Q29", "Q30"]}
    survey_subset = filter_survey_subset(survey, subsets)
    expected_qnums = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q29", "Q30"]
    pd.testing.assert_series_equal(
        survey_subset["number"], pd.Series(expected_qnums, name="number")
    )


def test_mark_is_scale_flipped():
    responses = [f"response{i}" for i in range(10)]
    is_flipped = mark_is_scale_flipped(responses)
    assert is_flipped == [False, True] * 5
