import json
import os

import pandas as pd

from src.analysis.io import create_subdirectory
from src.simulation.experiment import Experiment
from src.prompting.messages import (
    Survey,
    extract_user_prompts_from_survey_grouped,
    extract_user_prompts_from_survey_individual,
)


def load_survey(
    experiment: Experiment, question_format: str, is_reverse: bool
) -> Survey:

    survey_df = pd.read_csv(
        os.path.join(
            experiment.files["directory"], "variables", experiment.files["variables"]
        )
    )
    with open(
        os.path.join(
            experiment.files["directory"], "variables", experiment.files["subset"]
        ),
        "r",
    ) as f:
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
    simulated_survey: dict[str, dict], directory: str, experiment_name: str
):
    results_directory = create_subdirectory(
        os.path.join(directory, "results"), experiment_name
    )
    filename = f"{experiment_name}-results.json"
    with open(os.path.join(results_directory, filename), "w") as f:
        json.dump(simulated_survey, f)
        print(f"Successfully saved simulated responses as {filename}!")
