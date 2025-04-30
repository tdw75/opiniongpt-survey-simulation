import pytest

from src.simulation.models import is_phi_model


@pytest.mark.parametrize(
    "model_id, exp",
    [
        ("unsloth/Phi-3-mini-4k-instruct", True),
        ("some-name/phi-some-info", True),
        ("meta-llama/Meta-Llama-3-8B-Instruct", False),
    ],
)
def test_is_phi_model(model_id, exp):
    assert is_phi_model(model_id) == exp
