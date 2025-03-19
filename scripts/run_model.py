import json
import os
import sys

import fire
import pandas as pd

print(sys.path)
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from src.prompting.messages import build_messages
from src.simulation.inference import simulate_whole_survey
from src.simulation.models import load_opinion_gpt, load_llama


LOAD_MODEL = {
    "opinion_gpt": load_opinion_gpt,
    "llama": load_llama,
}

def huggingface_login() -> None:
    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv()
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print("Successfully logged in to Hugging Face!")
    else:
        print("Token is not set. Please save a token in the .env file.")

def main(model_name: str, directory: str, device: str = "cuda:2"):

    n_respondents = 1000
    by = "questions"

    model, tokenizer = LOAD_MODEL[model_name](device)
    survey_questions = load_survey(directory)
    respondents = simulate_whole_survey(model, tokenizer, survey_questions, by=by)
    survey_run = {
        "metadata": {"model_name": model_name, "by": by},  # todo: add rest of metadata
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
    huggingface_login()
    fire.Fire(main)
