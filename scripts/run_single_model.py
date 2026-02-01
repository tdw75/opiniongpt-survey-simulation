import os
import sys
from typing import Literal

import fire

print(sys.path)
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from src.simulation.inference import run_single
from src.simulation.models import ModelConfig, load_model, change_subgroup
from src.simulation.experiment import generate_run_id, huggingface_login
from src.simulation.survey import load_survey, save_results


def main(
    base_model_name: str = "phi",
    directory: str = "../data_files",
    subgroup: str = None,
    is_lora: bool = False,
    sample_size: int = 1000,
    batch_size: int = 50,
    filename: str = "variables.csv",
    subset_file: str = "final_subset.json",
    decoding_style: Literal["constrained", "unconstrained"] = "unconstrained",
    simulation_name: str = None,
    question_format: str = "individual",
    device: str = "cuda:2",
    **kwargs,  # LLM hyperparams
):
    survey_questions = load_survey(
        directory, filename, question_format, subset_file, False
    )
    survey_flipped = load_survey(
        directory, filename, question_format, subset_file, True
    )

    config = ModelConfig(
        base_model_name=base_model_name,
        subgroup=subgroup,
        is_lora=is_lora,
        is_persona=False,  # todo: parametrise
        device=device,
        aggregation_by="questions",  # todo: parametrise
        sample_size=sample_size,
        batch_size=batch_size,
        decoding_style=decoding_style,
        hyperparams=kwargs,
    )
    run_id = generate_run_id(base_model_name)
    model, tokenizer = load_model(config)
    model, config = change_subgroup(model, config, subgroup)

    survey_run = run_single(model, tokenizer, config, survey_questions, survey_flipped, run_id)
    save_results({config.run_name: survey_run}, directory, run_id, simulation_name)


if __name__ == "__main__":
    huggingface_login()
    fire.Fire(main)
