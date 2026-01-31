import os
import sys

import pandas as pd
import fire

print(sys.path)
print("Current working directory:", os.getcwd())
sys.path.append(os.getcwd())

from src.analysis.results import (
    survey_results_to_df,
    load_survey_results,
    load_survey_results_batch,
    survey_results_to_df_batch,
)


def main(directory: str, file_root: str):
    """Convert survey results from JSON to CSV format for cleaning."""
    simulation_directory = os.path.join(directory, "results", file_root)
    variables = pd.read_csv(os.path.join(directory, "variables", "variables.csv"))
    is_folder = os.path.isdir(os.path.join(simulation_directory, file_root))
    if is_folder:
        results = load_survey_results_batch(file_root, simulation_directory)
        df = survey_results_to_df_batch(results, variables)
    else:
        results = load_survey_results(f"{file_root}.json", simulation_directory)
        df = survey_results_to_df(results, variables)

    df.to_csv(os.path.join(simulation_directory, f"{file_root}.csv"))


if __name__ == "__main__":
    fire.Fire(main)
