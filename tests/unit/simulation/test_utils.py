import json

import pytest

from src.simulation.utils import get_nth_newest_file, print_results


@pytest.mark.parametrize("idx, expected", [(0, "20250429_results.json"), (1, "20250428_results.json")])
def test_get_nth_newest_file(idx, expected):
    assert get_nth_newest_file(idx, "test_data_files") == f"test_data_files/results/{expected}"


def test_print_results():
    results = json.load(open("test_data_files/results/20250429_results.json"))
    print_results(results)
