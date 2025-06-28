import logging
from timeit import default_timer as timer

from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.prompting.messages import Survey
from src.simulation.inference import simulate_whole_survey
from src.simulation.models import ModelConfig
from src.simulation.utils import mark_is_scale_flipped


def run_single(
    model: PeftModel | PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey_questions: Survey,
    survey_flipped: Survey,
    run_id: str,
):
    start = timer()
    logging.debug(model)
    outputs = simulate_whole_survey(
        model, tokenizer, config, survey_questions, survey_flipped
    )
    end = timer()
    return {
        "metadata": {
            "run_id": run_id,
            "execution_time": end - start,
            **config.model_dump(),
        },
        "questions": [prompt for prompt, _ in survey_questions],
        "choices": [ch for _, ch in survey_questions],
        "questions_flipped": [prompt for prompt, _ in survey_flipped],
        "choices_flipped": [ch for _, ch in survey_flipped],
        "outputs": outputs,
        "is_scale_flipped": {
            num: mark_is_scale_flipped(resp) for num, resp in outputs.items()
        },
    }
