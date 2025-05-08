import os
import sys

import fire


print(sys.path)
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from src.prompting.system import build_survey_context_message, build_persona_message
from src.simulation.inference import simulate_whole_survey
from src.simulation.models import load_model, MODEL_DIRECTORY, ModelConfig
from src.simulation.utils import (
    huggingface_login,
    load_survey,
    save_results,
    generate_run_id,
    get_single_question,
    print_results,
)


def main(
    base_model_name: str,
    directory: str,
    subgroup: str,
    is_lora: bool,
    filename: str = "variables.csv",
    question_format: str = "individual",
    device: str = "cuda:2",
    question_num: int = 0,
    **kwargs,
):
    config = ModelConfig(
        base_model_name=base_model_name,
        subgroup=subgroup,
        is_lora=is_lora,
        is_persona=False,  # todo: parametrise
        device=device,
        aggregation_by="questions",  # todo: parametrise
    )
    run_id = generate_run_id(base_model_name)
    survey_run = run_single(
        config,
        directory,
        run_id,
        filename,
        question_format,
        question_num,
        **kwargs,
    )
    print_results(survey_run)
    save_results(survey_run, directory, run_id)


def run_single(
    config: ModelConfig,
    directory: str,
    run_id: str,
    filename: str = "variables.csv",
    question_format: str = "individual",
    question_num: int = 0,
    **kwargs,
):
    model, tokenizer = load_model(config)
    print(model)

    survey_questions = load_survey(directory, filename, question_format)
    survey_questions = get_single_question(
        survey_questions, question_num
    )  # todo: delete after debugging
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


if __name__ == "__main__":
    huggingface_login()
    fire.Fire(main)
