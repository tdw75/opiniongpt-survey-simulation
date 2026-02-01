import os
import sys

import fire

print(sys.path)
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from src.simulation.inference import run_single
from src.simulation.models import ModelConfig, load_model, change_subgroup
from src.simulation.experiment import (
    generate_run_id,
    huggingface_login,
    load_experiment,
)
from src.simulation.survey import load_survey, save_results


def main(
    experiment_name: str,
    subgroup: str = None,
    is_lora: bool = False,
    question_format: str = "individual",
    device: str = "cuda:2",
    root_directory: str = "",
    **kwargs,  # additional LLM hyperparams
):
    experiment = load_experiment(experiment_name, root_directory)

    survey_questions = load_survey(experiment, question_format, False)
    survey_flipped = load_survey(experiment, question_format, True)

    config = ModelConfig(
        **experiment.simulation,
        subgroup=subgroup,
        is_lora=is_lora,
        is_persona=False,  # todo: parametrise
        device=device,
        aggregation_by="questions",  # todo: parametrise
        hyperparams=kwargs,
    )
    run_id = generate_run_id(experiment.simulation["base_model_name"])
    model, tokenizer = load_model(config)
    model, config = change_subgroup(model, config, subgroup)

    survey_run = run_single(
        model, tokenizer, config, survey_questions, survey_flipped, run_id
    )
    save_results(
        {config.run_name: survey_run},
        experiment.files["directory"],
        run_id,
        experiment_name,
    )


if __name__ == "__main__":
    huggingface_login()
    fire.Fire(main)
