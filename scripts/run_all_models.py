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
    phi_instruct_surveys = run_phi_instruct(
        survey_questions, base_model_name, device, count, run_id, **kwargs
    )
    opinion_gpt_surveys = run_opinion_gpt(
        survey_questions, base_model_name, device, count, run_id, **kwargs
    )  # todo: maybe add persona prompting for opinion gpt
    save_results(
        {**phi_instruct_surveys, **opinion_gpt_surveys},
        directory,
        run_id,
        simulation_name,
    )


def run_phi_instruct(
    survey_questions,
    base_model_name: str,
    device: str,
    count: int,
    run_id: str,
    **kwargs,
):
    simulated_surveys = {}
    config = ModelConfig(
        base_model_name=base_model_name,
        subgroup=None,
        is_lora=False,
        is_persona=True,
        device=device,
        aggregation_by="questions",
        count=count,
        hyperparams=kwargs,
    )
    model, tokenizer = load_model(config)

    for subgroup in adapters + [None]:

        model, config = change_subgroup(model, config, subgroup)
        run_name = get_run_name(config)
        simulated_surveys[run_name] = run_single(
            model, tokenizer, config, survey_questions, run_id
        )
    return simulated_surveys


def run_opinion_gpt(
    survey_questions,
    base_model_name: str,
    device: str,
    count: int,
    run_id: str,
    **kwargs,
):
    simulated_surveys = {}
    config = ModelConfig(
        base_model_name=base_model_name,
        is_lora=True,
        is_persona=False,
        device=device,
        aggregation_by="questions",
        count=count,
        hyperparams=kwargs,
    )
    model, tokenizer = load_model(config)

    for subgroup in adapters:
        run_name = get_run_name(config)
        model, config = change_subgroup(model, config, subgroup)
        simulated_surveys[run_name] = run_single(
            model, tokenizer, config, survey_questions, run_id
        )
    return simulated_surveys


if __name__ == "__main__":
    huggingface_login()
    fire.Fire(main)
