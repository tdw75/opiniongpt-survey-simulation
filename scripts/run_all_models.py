import os
import sys

import fire

print(sys.path)
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from src.simulation.models import adapters, ModelConfig, load_model, change_subgroup
from src.simulation.run import run_single
from src.simulation.utils import (
    huggingface_login,
    generate_run_id,
    save_results,
    get_run_name,
    load_survey,
)


def main(
    base_model_name: str = "phi",
    directory: str = "data_files",
    filename: str = "variables.csv",
    subset_file: str = None,
    simulation_name: str = None,
    question_format: str = "individual",
    device: str = "cuda:2",
    count: int = 500,
    **kwargs,  # LLM hyperparams
):
    survey_questions = load_survey(directory, filename, question_format, subset_file)
    run_id = generate_run_id(base_model_name)
    simulated_surveys = {}

    # BASE (NON-LORA) MODELS
    instruct_config = ModelConfig(
        base_model_name=base_model_name,
        subgroup=None,
        is_lora=False,
        is_persona=True,
        device=device,
        aggregation_by="questions",
        count=count,
        hyperparams=kwargs,
    )
    instruct_model, instruct_tokenizer = load_model(instruct_config)

    for subgroup in adapters + [None]:

        instruct_model, instruct_config = change_subgroup(
            instruct_model,
            instruct_config,
            subgroup,
        )
        run_name = get_run_name(base_model_name, False, subgroup)
        simulated_surveys[run_name] = run_single(
            instruct_model,
            instruct_tokenizer,
            instruct_config,
            survey_questions,
            run_id,
            **kwargs,
        )

    # OPINION GPT MODELS # todo: maybe add persona prompting for opinion gpt
    opinion_gpt_config = ModelConfig(
        base_model_name=base_model_name,
        is_lora=True,
        is_persona=False,
        device=device,
        aggregation_by="questions",
        count=count,
        hyperparams=kwargs,
    )
    opinion_gpt_model, opinion_gpt_tokenizer = load_model(opinion_gpt_config)

    for subgroup in adapters:

        run_name = get_run_name(base_model_name, True, subgroup)
        opinion_gpt_model, opinion_gpt_config = change_subgroup(
            opinion_gpt_model, opinion_gpt_config, subgroup
        )
        simulated_surveys[run_name] = run_single(
            opinion_gpt_model,
            opinion_gpt_tokenizer,
            opinion_gpt_config,
            survey_questions,
            run_id,
            **kwargs,
        )

    save_results(simulated_surveys, directory, run_id, simulation_name)


if __name__ == "__main__":
    huggingface_login()
    fire.Fire(main)
