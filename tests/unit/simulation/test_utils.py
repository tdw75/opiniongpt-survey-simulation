import pytest

from src.simulation.utils import get_run_name


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
