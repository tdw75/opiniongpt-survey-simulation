import json

import pytest

from src.simulation.utils import get_nth_newest_file, print_results_single, get_run_name


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
