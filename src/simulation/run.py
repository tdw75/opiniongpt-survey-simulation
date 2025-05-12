import logging

from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.prompting.system import build_survey_context_message
from src.simulation.inference import simulate_whole_survey
from src.simulation.models import ModelConfig


def run_single(
    model: PeftModel | PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey_questions: dict[str, str],
    run_id: str,
    number: int,
    **kwargs,
):
    logging.debug(model)
    system_prompt = build_survey_context_message()
    responses = simulate_whole_survey(
        model, tokenizer, config, survey_questions, system_prompt, number
    )
    return {  # todo: add rest of metadata, unpack all config values
        "metadata": {
            "model_id": config.model_id,
            "model_type": config.model_type,
            "subgroup": (
                config.subgroup if config.is_lora else "none"
            ),  # todo: update after adding persona prompting
            "system_prompt": system_prompt,
            "aggregation_by": config.aggregation_by,
            "run_id": run_id,
            **kwargs,
        },
        "questions": survey_questions,
        "responses": responses,
    }
