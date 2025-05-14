import json

import pandas as pd
import pytest

from src.analysis.results import (
    get_nth_newest_file,
    print_results_single,
    survey_results_to_df,
)


def test_survey_results_to_df():
    survey_results = json.load(open("test_data_files/results/20250428_results.json"))
    df = survey_results_to_df({"phi-aus": survey_results})
    expected = {
        "number": ["Q1"] * 2 + ["Q2"] * 2,
        "question": ["I believe in Santa Claus"] * 2
        + ["I believe in the Tooth Fairy"] * 2,
        "response": ["1: agree", "2: disagree", "1: agree", "1: agree"],
        "model_id": ["llama-3"] * 4,
        "run_id": ["20250428"] * 4,
        "system_prompt": ["you are a survey participant"] * 4,
    }
    pd.testing.assert_frame_equal(df, pd.DataFrame(expected))


@pytest.mark.parametrize(
    "idx, expected", [(0, "20250429_results.json"), (1, "20250428_results.json")]
)
def test_get_nth_newest_file(idx, expected):
    assert (
        get_nth_newest_file(idx, "test_data_files")
        == f"test_data_files/results/{expected}"
    )


def test_print_results():
    results = json.load(open("test_data_files/results/20250429_results.json"))
    print_results_single(results, "title")
