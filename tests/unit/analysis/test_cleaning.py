import pytest

from src.analysis.cleaning import strip_response


@pytest.mark.parametrize(
    "string, output",
    [
        ("Your response: 1: value", "1: value"),
        ("Response:   9: 9", "9: 9"),
        ("Your response: 10: value value", "10: value value"),
    ],
)
def test_strip_response(string, output):
    assert strip_response(string) == output