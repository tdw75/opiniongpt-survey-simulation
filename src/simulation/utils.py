import json
import os

import pandas as pd

from messages import build_messages


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


def load_survey(directory: str, file_name: str) -> dict[str, str]:
    survey_df = pd.read_csv(os.path.join(directory, file_name))
    survey = build_messages(survey_df)
    print("Successfully loaded survey!")
    return survey


def save_survey(simulated_survey: dict[str, dict], directory: str):
    with open(os.path.join(directory, "survey_run.json"), "w") as f:
        json.dump(simulated_survey, f)
        print("Successfully saved simulated responses!")
