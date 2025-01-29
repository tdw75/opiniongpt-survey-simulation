from demographics.base import pascal_to_snake


def test_pascal_to_snake():
    assert pascal_to_snake("PascalCase") == "pascal_case"
