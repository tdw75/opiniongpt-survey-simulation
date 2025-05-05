import glob
import json
import os
from datetime import datetime


import pandas as pd

from src.prompting.messages import extract_user_prompts_from_survey_individual
from src.prompting.messages import extract_user_prompts_from_survey_grouped


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


def load_survey(directory: str, file_name: str, question_format: str) -> dict[str, str]:
    survey_df = pd.read_csv(os.path.join(directory, file_name))
    if question_format == "grouped":
        survey = extract_user_prompts_from_survey_grouped(survey_df)
    elif question_format == "individual":
        survey = extract_user_prompts_from_survey_individual(survey_df)
    else:
        raise ValueError(f"Invalid question format: {question_format}")
    print("Successfully loaded survey!")
    return survey


def save_results(simulated_survey: dict[str, dict], directory: str, run_id: str):
    results_directory = os.path.join(directory, "results")
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    with open(os.path.join(results_directory, f"{run_id}.json"), "w") as f:
        json.dump(simulated_survey, f)
        print("Successfully saved simulated responses!")


def generate_run_id(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%Hh%M.%S")
    return f"{timestamp}-{model_name}"


def get_single_question(survey: dict[str, str], idx: int = 0) -> dict[str, str]:
    """debugging function: selects a single question from the survey"""
    single_question = list(survey.items())[idx]
    return {single_question[0]: single_question[1]}


def get_nth_newest_file(n: int, directory: str):
    files_path = os.path.join(directory, "results/*")
    files = sorted(glob.iglob(files_path), key=os.path.getmtime, reverse=True)
    return files[n]


def print_results(results: dict[str, dict]):
    print("=" * 20, "METADATA", "=" * 20)
    for k, v in results["metadata"].items():
        print(f"{k}: {v}")
    print("=" * 20, "RESULTS", "=" * 20)
    for num, question in results["questions"].items():
        print(f"{question}")
        for i, response in enumerate(results["responses"][num]):
            print(f"* {i}. {response}")
