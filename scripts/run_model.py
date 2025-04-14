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
from src.simulation.utils import (
    huggingface_login,
    load_survey,
    save_survey,
    generate_run_id,
)

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
    question_format: str = "individual",
    device: str = "cuda:2",
    **kwargs,
):

    by = "questions"  # todo: parametrise
    # todo: separate model loading from inference (maybe loop through subgroups)
    model, tokenizer = LOAD_MODEL[model_name](device)
    model = CHANGE_SUBGROUP[model_name](model, subgroup)

    print(model)
    survey_questions = load_survey(directory, filename, question_format)
    respondents = simulate_whole_survey(
        model, tokenizer, survey_questions, by, hyperparams=kwargs
    )
    survey_run = {  # todo: add rest of metadata
        "metadata": {"model_name": model_name, "by": by, **kwargs},
        "respondents": respondents,
    }
    save_survey(survey_run, directory, run_id=generate_run_id(model_name))


if __name__ == "__main__":
    huggingface_login()
    fire.Fire(main)
