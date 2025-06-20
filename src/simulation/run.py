import logging
from timeit import default_timer as timer

from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.simulation.inference import simulate_whole_survey
from src.simulation.models import ModelConfig
from src.simulation.utils import mark_is_scale_flipped


def run_single(
    model: PeftModel | PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey_questions: dict[str, str],
    survey_flipped: dict[str, str],
    run_id: str,
):
    start = timer()
    logging.debug(model)
    responses = simulate_whole_survey(
        model, tokenizer, config, survey_questions, survey_flipped
    )
    end = timer()
    return {
        "metadata": {
            "run_id": run_id,
            "execution_time": end - start,
            **config.model_dump(),
        },
        "questions": survey_questions,
        "questions_flipped": survey_flipped,
        "responses": responses,
        "is_scale_flipped": {
            num: mark_is_scale_flipped(resp) for num, resp in responses.items()
        },
    }
