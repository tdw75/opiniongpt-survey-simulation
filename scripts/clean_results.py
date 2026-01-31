import ast
import json
import os

import pandas as pd

from src.analysis.cleaning import (
    pipeline_clean_generated_responses,
    remap_response_keys,
)
from src.analysis.invalid_responses import pipeline_identify_invalid_responses
from src.data.variables import responses_to_map, ResponseMap, QNum
from src.demographics.config import category_to_question


def main(file_root: str, directory: str = "../data_files" ):

    results_directory = os.path.join(directory, "results", file_root)
    df = pd.read_csv(os.path.join(results_directory, f"{file_root}.csv"), index_col=0)
    variables = pd.read_csv(
        os.path.join(directory, "variables", "variables.csv"), index_col=0
    )

    responses, responses_flipped = load_response_maps(variables)
    df = pipeline_clean_generated_responses(df)
    df = pipeline_identify_invalid_responses(df, responses, responses_flipped)
    df = remap_response_keys(df, "final_response")
    df.to_csv(os.path.join(results_directory, f"{file_root}-clean.csv"))

    reasons = (
        df["reason_invalid"].value_counts(normalize=True).sort_values(ascending=False)
    )
    reasons.to_json(
        os.path.join(results_directory, f"{file_root}_invalid_summary.json")
    )
    qnums = set(df["number"])
    cat_counts = {
        c: len(qnums.intersection(q)) for c, q in category_to_question.items()
    }
    with open(
        os.path.join(results_directory, f"{file_root}-category-counts.json"), "w"
    ) as f:
        json.dump(cat_counts, f)


def load_response_maps(
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


if __name__ == "__main__":
    import fire

    fire.Fire(main)
