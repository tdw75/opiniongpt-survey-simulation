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
):
    start = timer()
    logging.debug(model)
    responses = simulate_whole_survey(model, tokenizer, config, survey_questions)
    end = timer()
    return {
        "metadata": {
            "run_id": run_id,
            "execution_time": end - start,
            **config.model_dump(),
        },
        "questions": survey_questions,
        "responses": responses,
    }
