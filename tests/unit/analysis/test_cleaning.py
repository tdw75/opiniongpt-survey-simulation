import pytest

from src.analysis.cleaning import strip_response, split_response_with_missing


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


@pytest.mark.parametrize(
    "response, expected", [
        ("1.- Very important", (1, "Very important")),
        ("1: Very important", (1, "Very important")),
        ("1:  Very important", (1, "Very important")),
        ("9: 9", (9, "9")),
        ("1: Very important 2: Rather important", (-1, "missing")),
    ]
)
def test_split_response_string(response, expected):
    split = split_response_with_missing(response)
    assert split == expected