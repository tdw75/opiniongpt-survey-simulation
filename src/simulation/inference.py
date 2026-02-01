import logging
from timeit import default_timer as timer

from peft import PeftModel
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.prompting.messages import Survey
from src.simulation.decoders import (
    ConstrainedDecoder,
    UnconstrainedDecoder,
    BaseDecoder,
)
from src.simulation.models import ModelConfig
from src.utils import mark_is_scale_flipped

logger = logging.getLogger(__name__)


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
        "questions": extract_prompts(survey_questions),
        "choices": extract_choices(survey_questions),
        "questions_flipped": extract_prompts(survey_flipped),
        "choices_flipped": extract_choices(survey_flipped),
        "responses": outputs,
        "is_scale_flipped": {
            num: mark_is_scale_flipped(resp) for num, resp in outputs.items()
        },
    }


def simulate_whole_survey(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: Survey,
    flipped: Survey,
) -> dict[str, list[str]]:
    logger.debug(model)
    decoder = get_decoder(model, tokenizer, config)
    responses: dict[str, list[str]] = {}
    for qnum, question in tqdm(survey.items(), desc=decoder.config.run_name):
        responses[qnum] = decoder.simulate_question(qnum, survey[qnum], flipped[qnum])
    return responses


def get_decoder(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: ModelConfig
) -> BaseDecoder:
    if config.decoding_style == "constrained":
        return ConstrainedDecoder(model, tokenizer, config)
    elif config.decoding_style == "unconstrained":
        return UnconstrainedDecoder(model, tokenizer, config)
    else:
        raise ValueError(f"Unknown decoding style: {config.decoding_style}")


def extract_prompts(survey: Survey) -> dict[str, str]:
    return {qnum: prompt for qnum, (prompt, _) in survey.items()}


def extract_choices(survey: Survey) -> dict[str, list[str]]:
    return {qnum: choices for qnum, (_, choices) in survey.items()}
