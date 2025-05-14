import os
import sys

import fire

print(sys.path)
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from src.simulation.models import adapters, ModelConfig, load_model
from src.simulation.run import run_single
from src.simulation.utils import (
    huggingface_login,
    generate_run_id,
    save_results,
    get_run_name, load_survey,
)


def main(
    base_model_name: str = "phi",
    directory: str = "data_files",
    filename: str = "variables.csv",
    subset_file: str = None,
    question_format: str = "individual",
    device: str = "cuda:2",
    number: int = 1000,
    **kwargs,  # LLM hyperparams
):
    survey_questions = load_survey(directory, filename, question_format, subset_file)

    instruct_config = ModelConfig(
        base_model_name=base_model_name,
        subgroup=None,
        is_lora= False,
        is_persona=False,
        device=device,
        aggregation_by="questions",
        hyperparams=kwargs,
    )
    instruct_model, instruct_tokenizer = load_model(instruct_config)
    run_id = generate_run_id(base_model_name)
    run_name = get_run_name(base_model_name, False, None)
    simulated_surveys = {  # todo: add personas
        run_name: run_single(
            instruct_model,
            instruct_tokenizer,
            instruct_config,
            survey_questions,
            run_id,
            number,
            **kwargs
        )
    }

    opinion_gpt_config = ModelConfig(
        base_model_name=base_model_name,
        is_lora=True,
        is_persona=False,
        device=device,
        aggregation_by="questions",
        hyperparams=kwargs,
    )
    opiniongpt_model, opiniongpt_tokenizer = load_model(opinion_gpt_config)
    for subgroup in adapters:
        run_name = get_run_name(base_model_name, True, subgroup)
        opinion_gpt_config.change_subgroup(subgroup)
        simulated_surveys[run_name] = run_single(
            opiniongpt_model,
            opiniongpt_tokenizer,
            opinion_gpt_config,
            survey_questions,
            run_id,
            number,
            **kwargs
        )

    save_results(simulated_surveys, directory, run_id)


if __name__ == "__main__":
    huggingface_login()
    fire.Fire(main)
