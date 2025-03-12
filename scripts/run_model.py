import json
import os
import sys

import fire
import pandas as pd

from prompting.messages import build_messages
from src.simulation.inference import simulate_set_of_responses_single_question
from src.simulation.models import load_opinion_gpt, load_llama, load_mock_model

print(sys.path)


def main(model_name: str, directory: str, device: str = "cuda:2"):

    n_respondents = 1000

    if model_name == "opinion_gpt":
        model, tokenizer = load_opinion_gpt(device)
    elif model_name == "llama":
        model, tokenizer = load_llama(device)
    elif model_name == "mock":
        model, tokenizer = load_mock_model(device)
        print("Running Mock Model")
        n_respondents = 2
    else:
        raise ValueError(f"Model {model_name} not found")

    survey_questions = load_survey(directory)
    respondents = simulate_set_of_responses_single_question(model, tokenizer, n_respondents, directory)
    survey_run = {
        "metadata": {"model_name": model_name},  # todo: add rest of metadata
        "respondents": respondents,
    }
    save_survey(survey_run, directory)


def load_survey(directory: str) -> dict[str, str]:
    survey = pd.read_csv(os.path.join(directory, "variables.csv"))
    return build_messages(survey)


def save_survey(simulated_survey: dict[str, dict], directory: str):
    with open(os.path.join(directory, "survey_run.json"), "w") as f:
        json.dump(simulated_survey, f)


if __name__ == "__main__":
    fire.Fire(main)
