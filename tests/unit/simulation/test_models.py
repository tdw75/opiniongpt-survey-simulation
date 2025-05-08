import pytest

from src.simulation.models import ModelConfig


class TestModelConfig:

    def test_initialisation(self):

        config = ModelConfig(
            base_model_name="phi",
            subgroup=None,
            is_lora=True,
            is_persona=False,
            device="cuda:2",
            aggregation_by="questions",
            hyperparams={"new_param": True, "max_new_tokens": 9999},
        )

        assert config.base_model_name == "phi"
        assert config.model_id == "unsloth/Phi-3-mini-4k-instruct"
        assert config.model_type == "OpinionGPT"
        assert config.hyperparams == {
            "min_new_tokens": 4,
            "max_new_tokens": 9999,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.8,
            "new_param": True,
        }

    @pytest.mark.parametrize("model_name, exp", [("phi", True), ("llama", False)])
    def test_is_phi_model(self, model_name, exp):
        config = ModelConfig(base_model_name=model_name)
        assert config.is_phi_model == exp

    def test_change_subgroup(self):
        config = ModelConfig(subgroup="german")
        config.change_subgroup("american")
        assert config.subgroup == "american"
