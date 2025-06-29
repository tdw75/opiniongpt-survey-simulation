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
        "questions": extract_prompts(survey_questions),
        "choices": extract_choices(survey_questions),
        "questions_flipped": extract_prompts(survey_flipped),
        "choices_flipped": extract_choices(survey_flipped),
        "responses": outputs,
        "is_scale_flipped": {
            num: mark_is_scale_flipped(resp) for num, resp in outputs.items()
        },
    }


def extract_prompts(survey: Survey) -> dict[str, str]:
    return {qnum: prompt for qnum, (prompt, _) in survey.items()}


def extract_choices(survey: Survey) -> dict[str, list[str]]:
    return {qnum: choices for qnum, (_, choices) in survey.items()}
