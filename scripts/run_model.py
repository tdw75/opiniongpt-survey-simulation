import os
import sys

import fire

print(sys.path)
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from src.simulation.inference import simulate_whole_survey
from src.simulation.models import (
    load_opinion_gpt,
    load_llama,
    change_adapter,
    change_persona,
)
from src.simulation.utils import huggingface_login, load_survey, save_survey


LOAD_MODEL = {
    "opinion_gpt": load_opinion_gpt,
    "llama": load_llama,
}
CHANGE_SUBGROUP = {
    "opinion_gpt": change_adapter,
    "llama": change_persona,
}


def main(
    model_name: str,
    directory: str,
    subgroup: str,
    filename: str = "variables.csv",
    device: str = "cuda:2",
):

    n_respondents = 1000
    by = "questions"  # todo: parametrise

    # todo: separate model loading from inference (maybe loop through subgroups)
    model, tokenizer = LOAD_MODEL[model_name](device)
    model = CHANGE_SUBGROUP[model_name](model, subgroup)
    survey_questions = load_survey(directory, filename)
    respondents = simulate_whole_survey(model, tokenizer, survey_questions, by=by)
    survey_run = {
        "metadata": {"model_name": model_name, "by": by},  # todo: add rest of metadata
        "respondents": respondents,
    }
    save_survey(survey_run, directory)


if __name__ == "__main__":
    huggingface_login()
    fire.Fire(main)
