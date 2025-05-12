import logging

from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.prompting.system import build_survey_context_message
from src.simulation.inference import simulate_whole_survey
from src.simulation.models import ModelConfig, load_model
from src.simulation.utils import load_survey, get_single_question


def run_single(
    model: PeftModel | PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    directory: str,
    run_id: str,
    filename: str = "variables.csv",
    question_format: str = "individual",
):
    logging.debug(model)

    survey_questions = load_survey(directory, filename, question_format)
    system_prompt = build_survey_context_message()
    responses = simulate_whole_survey(
        model, tokenizer, config, survey_questions, system_prompt
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
