import json
import os
from datetime import datetime

import pandas as pd

from src.analysis.visualisations import RENAME_MAP, reformat_index
from src.data.variables import QNum, ResponseMap, remap_response_maps
from src.demographics.config import dimensions
from src.prompting.messages import (
    extract_user_prompts_from_survey_grouped,
    extract_user_prompts_from_survey_individual,
    Survey,
)
from src.simulation.models import ModelName, adapters


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


def load_survey(
    directory: str,
    file_name: str,
    question_format: str,
    subset_name: str,
    is_reverse: bool,
) -> Survey:

    survey_df = pd.read_csv(os.path.join(directory, "variables", file_name))
    if subset_name:
        with open(os.path.join(directory, "variables", subset_name), "r") as f:
            subsets = json.load(f)
        survey_df = filter_survey_subset(survey_df, subsets)

    if question_format == "grouped":
        survey = extract_user_prompts_from_survey_grouped(survey_df, is_reverse)
    elif question_format == "individual":
        survey = extract_user_prompts_from_survey_individual(
            survey_df, False, is_reverse
        )
    else:
        raise ValueError(f"Invalid question format: {question_format}")

    print(
        f"Successfully loaded survey in {'reverse' if is_reverse else 'normal'} order!"
    )
    return survey


def filter_survey_subset(survey: pd.DataFrame, subsets: dict) -> pd.DataFrame:
    group_mask = survey["group"].isin(subsets.get("groups", []))
    questions_mask = survey["number"].isin(subsets.get("individual_questions", []))
    return survey[group_mask | questions_mask].reset_index(drop=True)


def save_results(
    simulated_survey: dict[str, dict], directory: str, run_id: str, simulation_name: str
):
    if simulation_name:
        results_directory = os.path.join(directory, "results", simulation_name)
    else:
        results_directory = os.path.join(directory, "results")
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    with open(os.path.join(results_directory, f"{run_id}.json"), "w") as f:
        json.dump(simulated_survey, f)
        print("Successfully saved simulated responses!")


def generate_run_id(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%Hh%M.%S")
    return f"{timestamp}-{model_name}"


def mark_is_scale_flipped(responses: list[str]):
    """
    responses : list of generated responses for a single survey question
    """
    return [i % 2 == 1 for i in range(len(responses))]


def key_as_int(response_map: dict[QNum, ResponseMap]) -> dict:
    return {
        qnum: {int(k): v for k, v in resp.items()}
        for qnum, resp in response_map.items()
    }


def create_subdirectory(directory: str, subdirectory: str) -> str:
    """
    Create a subdirectory if it does not exist.
    """
    full_path = os.path.join(directory, subdirectory)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path


def load_response_maps(directory: str = "../data_files") -> dict[QNum, ResponseMap]:
    with open(
        os.path.join(directory, "variables/response_map_original.json"), "r"
    ) as f1:
        response_map = key_as_int(json.load(f1))
        response_map = remap_response_maps(response_map)
        response_map = {k: v for k, v in response_map.items() if k != "Q215"}

    return response_map


def save_latex_table(df: pd.DataFrame, directory: str, name: str, **kwargs):
    df = df.rename(columns=RENAME_MAP, errors="ignore")
    df.index = reformat_index(df.index)
    df.to_latex(os.path.join(directory, name), **kwargs)


def load_data_dict(
    filename: str,
    root_directory: str,
    models: list[ModelName],
    grouping: str = "subgroup",
):
    directory = os.path.join(root_directory, "results", filename, "data")
    path_template = os.path.join(directory, "{grouping}-{m}-{sg}-responses.csv")
    groups = adapters if grouping == "subgroup" else dimensions.keys()
    data = {
        sg: {
            m: pd.read_csv(
                path_template.format(grouping=grouping, m=m, sg=sg), index_col=0
            )
            for m in models
            if m != "base"
        }
        for sg in groups
    }
    if "base" in models:
        base = pd.read_csv(
            os.path.join(directory, f"{grouping}-base-responses.csv"), index_col=0
        )
        for sg in groups:
            data[sg]["base"] = base
    return data
