import logging
from timeit import default_timer as timer

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
    start = timer()
    logging.debug(model)
    system_prompt = build_survey_context_message()
    responses = simulate_whole_survey(
        model, tokenizer, config, survey_questions, system_prompt, number
    )
    end = timer()
    return {  # todo: add rest of metadata, unpack all config values
        "metadata": {
            **config.model_dump(),
            "system_prompt": system_prompt,
            "run_id": run_id,
            "execution_time": end - start,
            "num_questions": number,
            **kwargs,
        },
        "questions": survey_questions,
        "responses": responses,
    }
