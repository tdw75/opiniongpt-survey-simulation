import os
import sys

import fire


print(sys.path)
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from src.prompting.system import build_survey_context_message
from src.simulation.inference import simulate_whole_survey
from src.simulation.models import load_model, MODEL_DIRECTORY
from src.simulation.utils import (
    huggingface_login,
    load_survey,
    save_results,
    generate_run_id,
    get_single_question,
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

    by = "questions"  # todo: parametrise
    # todo: separate model loading from inference (maybe loop through subgroups)
    model, tokenizer = load_model(base_model_name, subgroup, is_lora, device)
    print(model)

    survey_questions = load_survey(directory, filename, question_format)
    survey_questions = get_single_question(survey_questions, question_num)  # todo: delete after debugging
    system_prompt = build_survey_context_message()
    respondents = simulate_whole_survey(
        model, tokenizer, survey_questions, by, system_prompt, hyperparams=kwargs
    )
    survey_run = {  # todo: add rest of metadata
        "metadata": {
            "model_id": MODEL_DIRECTORY[base_model_name],
            "model_type": "OpinionGPT" if is_lora else "instruct",
            "system_prompt": system_prompt,
            "by": by,
            "run_id": (run_id := generate_run_id(base_model_name)),
            **kwargs,
        },
        "questions": survey_questions,
        "respondents": respondents,
    }
    save_results(survey_run, directory, run_id)


if __name__ == "__main__":
    huggingface_login()
    fire.Fire(main)
