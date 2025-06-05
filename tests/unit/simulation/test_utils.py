import pandas as pd
import pytest

from src.simulation.utils import (
    filter_survey_subset,
    mark_is_scale_flipped,
    get_batches
)


def test_filter_survey_subset():
    survey = pd.read_csv("test_data_files/sample_variables.csv")
    subsets = {"groups": ["Important in life"], "individual_questions": ["Q29", "Q30"]}
    survey_subset = filter_survey_subset(survey, subsets)
    expected_qnums = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q29", "Q30"]
    pd.testing.assert_series_equal(
        survey_subset["number"], pd.Series(expected_qnums, name="number")
    )


@pytest.mark.parametrize("sample_size", (4, 5))
def test_get_batches(sample_size):
    expected_batch1 = [[{"message1": "content"}], [{"message2": "content"}]]
    expected_batch2 = [[{"message3": "content"}], [{"message4": "content"}]]
    expected = [expected_batch1, expected_batch2]

    if sample_size == 5:
        expected.append([[{"message5": "content"}]])

    messages_batched = [[{f"message{i}": "content"}] for i in range(1, sample_size + 1)]
    for i, batch in enumerate(get_batches(messages_batched, batch_size=2)):
        assert batch == expected[i]


def test_mark_is_scale_flipped():
    responses = [f"response{i}" for i in range(10)]
    is_flipped = mark_is_scale_flipped(responses)
    assert is_flipped == [False, True] * 5
