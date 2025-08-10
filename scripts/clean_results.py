import ast
import os

import pandas as pd

from src.analysis.cleaning import pipeline_clean_generated_responses
from src.analysis.invalid_responses import pipeline_identify_invalid_responses
from src.data.variables import responses_to_map, ResponseMap


def main(file_root: str, directory: str = "../data_files"):

    df = pd.read_csv(os.path.join(directory, "results", file_root, f"{file_root}.csv"), index_col=0)
    variables = pd.read_csv(os.path.join(directory, "variables", "variables.csv"), index_col=0)

    responses, responses_flipped = load_response_maps(variables)
    df = pipeline_clean_generated_responses(df)
    df = pipeline_identify_invalid_responses(df, responses, responses_flipped)
    df.to_csv(os.path.join(directory, "results", file_root, f"{file_root}-clean.csv"))

    reasons = df["reason_invalid"].value_counts(normalize=True).sort_values(ascending=False)
    reasons.to_json(os.path.join(directory, "results", file_root, f"{file_root}_invalid_summary.json"))
    print(reasons)

def load_response_maps(variables: pd.DataFrame) -> tuple[dict[str, ResponseMap], dict[str, ResponseMap]]:
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

if __name__ == "__main__":
    main("simulation-500-0_7-unconstrained")