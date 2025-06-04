import json
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
    load_survey,
)


def main(
    base_model_name: str = "phi",
    directory: str = "data_files",
    filename: str = "variables.csv",
    subset_file: str = "final_subset.json",
    simulation_name: str = None,
    question_format: str = "individual",
    device: str = "cuda:2",
    sample_size: int = 500,
    batch_size: int = None,
    run_id: str = None,
    is_few_shot: bool = False,
    **kwargs,  # LLM hyperparams
):

    run_id = run_id or generate_run_id(base_model_name)
    simulated_surveys = {}

    shared_config_vars = {
        "base_model_name": base_model_name,
        "sample_size": sample_size,
        "batch_size": batch_size or max(sample_size // 2, 50),
        "device": device,
        "is_few_shot": is_few_shot,
        "hyperparams": kwargs,
    }

    # todo: clear cache in loop
    survey_questions = load_survey(
        directory, filename, question_format, subset_file, False
    )
    survey_flipped = load_survey(
        directory, filename, question_format, subset_file, True
    )
    phi_instruct_surveys = run_phi_instruct(
        survey_questions, survey_flipped, shared_config_vars, run_id
    )
    opinion_gpt_surveys = run_opinion_gpt(
        survey_questions, survey_flipped, shared_config_vars, run_id
    )
    simulated_surveys.update(phi_instruct_surveys)
    simulated_surveys.update(opinion_gpt_surveys)

    save_results(
        simulated_surveys,
        directory,
        run_id,
        simulation_name,
    )


def run_phi_instruct(
    survey_questions: dict[str, str],
    survey_flipped: dict[str, str],
    shared_config_vars: dict,
    run_id: str,
):
    simulated_surveys = {}
    config = ModelConfig(
        **shared_config_vars,
        is_lora=False,
        is_persona=True,
        aggregation_by="questions",
    )
    model, tokenizer = load_model(config)

    for subgroup in adapters + [None]:
        model, config = change_subgroup(model, config, subgroup)
        simulated_surveys[config.run_name] = run_single(
            model, tokenizer, config, survey_questions, survey_flipped, run_id
        )
    return simulated_surveys


def run_opinion_gpt(
    survey_questions: dict[str, str],
    survey_flipped: dict[str, str],
    shared_config_vars: dict,
    run_id: str,
):
    # todo: add persona prompting for opinion gpt
    simulated_surveys = {}
    config = ModelConfig(
        **shared_config_vars,
        is_lora=True,
        is_persona=False,
        aggregation_by="questions",
    )
    model, tokenizer = load_model(config)

    for subgroup in adapters:
        model, config = change_subgroup(model, config, subgroup)
        simulated_surveys[config.run_name] = run_single(
            model, tokenizer, config, survey_questions, survey_flipped, run_id
        )
    return simulated_surveys


if __name__ == "__main__":
    huggingface_login()
    fire.Fire(main)
