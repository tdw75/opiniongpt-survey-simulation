import ast
import json
import os
import sys

import fire
import pandas as pd

print(sys.path)
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from src.analysis.cleaning import pipeline_clean_generated_responses, remap_response_keys
from src.analysis.invalid_responses import pipeline_identify_invalid_responses
from src.data.variables import responses_to_map, ResponseMap, QNum
from src.demographics.config import category_to_question


def main(simulation_name: str, directory: str = "data_files"):
    """
    Clean simulation results for analysis, identify invalid responses and
    remap response keys to original scale.
    """

    results_directory = os.path.join(directory, "results", simulation_name)
    df = pd.read_csv(os.path.join(results_directory, f"{simulation_name}-results.csv"), index_col=0)
    variables_directory = os.path.join(directory, "variables")
    variables = pd.read_csv(os.path.join(variables_directory, "variables.csv"), index_col=0)

    responses, responses_flipped = get_response_maps_from_variables(variables)
    save_response_maps(responses, variables_directory)
    df = pipeline_clean_generated_responses(df)
    df = pipeline_identify_invalid_responses(df, responses, responses_flipped)
    df = remap_response_keys(df, "final_response")
    df.to_csv(os.path.join(results_directory, f"{simulation_name}-clean.csv"))

    reasons = (
        df["reason_invalid"].value_counts(normalize=True).sort_values(ascending=False)
    )
    reasons.to_json(
        os.path.join(results_directory, f"{simulation_name}_invalid_summary.json")
    )
    qnums = set(df["number"])
    cat_counts = {
        c: len(qnums.intersection(q)) for c, q in category_to_question.items()
    }
    with open(
        os.path.join(results_directory, f"{simulation_name}-category-counts.json"), "w"
    ) as f:
        json.dump(cat_counts, f)


def get_response_maps_from_variables(
    variables: pd.DataFrame,
) -> tuple[dict[QNum, ResponseMap], dict[QNum, ResponseMap]]:
    responses = {}
    responses_flipped = {}

    for _, row in variables.iterrows():
        responses[row["number"]] = responses_to_map(
            ast.literal_eval(row["responses"]), is_scale_flipped=False
        )
        responses_flipped[row["number"]] = responses_to_map(
            ast.literal_eval(row["responses"]), is_scale_flipped=True
        )

    return responses, responses_flipped


def save_response_maps(responses: dict[QNum, ResponseMap], directory: str):
    with open(os.path.join(directory, "response_map_original.json"), "w") as f:
        json.dump(responses, f)


if __name__ == "__main__":
    fire.Fire(main)

