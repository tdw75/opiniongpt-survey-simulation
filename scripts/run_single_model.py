import os
import sys

import fire

print(sys.path)
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from src.simulation.run import run_single
from src.simulation.models import ModelConfig, load_model, change_subgroup
from src.simulation.utils import (
    huggingface_login,
    save_results,
    generate_run_id,
    get_run_name,
    load_survey,
)


def main(
    base_model_name: str,
    directory: str,
    subgroup: str,
    is_lora: bool,
    count: int = 1000,
    filename: str = "variables.csv",
    subset_file: str = None,
    simulation_name: str = None,
    question_format: str = "individual",
    device: str = "cuda:2",
    **kwargs,  # LLM hyperparams
):
    survey_questions = load_survey(directory, filename, question_format, subset_file)

    config = ModelConfig(
        base_model_name=base_model_name,
        subgroup=subgroup,
        is_lora=is_lora,
        is_persona=False,  # todo: parametrise
        device=device,
        aggregation_by="questions",  # todo: parametrise
        count=count,
        hyperparams=kwargs,
    )
    run_id = generate_run_id(base_model_name)
    model, tokenizer = load_model(config)
    model, config = change_subgroup(model, config, subgroup)

    survey_run = run_single(
        model, tokenizer, config, survey_questions, run_id, **kwargs
    )
    run_name = get_run_name(base_model_name, is_lora, subgroup if is_lora else None)
    save_results({run_name: survey_run}, directory, run_id, simulation_name)


if __name__ == "__main__":
    huggingface_login()
    fire.Fire(main)
