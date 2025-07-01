import os

import pandas as pd

from src.analysis.results import (
    survey_results_to_df,
    load_survey_results,
    load_survey_results_batch,
    survey_results_to_df_batch,
)


def main(directory: str, file_root: str):
    variables = pd.read_csv(os.path.join(directory, "variables", "variables.csv"))
    is_folder = os.path.isdir(os.path.join(directory, "results", file_root))
    if is_folder:
        results = load_survey_results_batch(file_root, directory)
        df = survey_results_to_df_batch(results, variables)
    else:
        results = load_survey_results(f"{file_root}.json", directory)
        df = survey_results_to_df(results, variables)

    df.to_csv(os.path.join(directory, "results", f"{file_root}.csv"))


if __name__ == "__main__":
    main("../data_files", "simulation-2-200")
