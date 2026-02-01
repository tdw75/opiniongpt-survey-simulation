import glob
import json
import os

import pandas as pd

from src.demographics.config import dimensions
from src.simulation.models import ModelName, adapters


def load_survey_results_batch(
    files_folder: str, directory: str
) -> list[dict[str, dict]]:
    results = []
    folder_path = os.path.join(directory, files_folder)
    for path in glob.glob(os.path.join(folder_path, "*.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


def load_survey_results(file_name: str, directory: str) -> dict[str, dict]:
    path = os.path.join(directory, file_name)
    with open(path) as f:
        return json.load(f)


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


def survey_results_to_df_batch(
    survey_results: list[dict], variables: pd.DataFrame
) -> pd.DataFrame:
    dfs = []
    for results in survey_results:
        dfs.append(survey_results_to_df(results, variables))
    df = pd.concat(dfs)
    return df.reset_index(drop=True)


def survey_results_to_df(
    survey_results: dict[str, dict], variables: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for model, results in survey_results.items():
        for num, responses in results["responses"].items():
            variable = variables[variables["number"] == num]
            for response, is_flipped in zip(
                responses, results["is_scale_flipped"][num]
            ):
                suffix = "_flipped" if is_flipped else ""
                row = {
                    "model": model,
                    "number": num,
                    "group": variable["group"].item(),
                    "subtopic": variable["subtopic"].item(),
                    "question": results[f"questions{suffix}"][num],
                    "choices": results[f"choices{suffix}"][num],
                    "response": response,
                    "is_scale_flipped": is_flipped,
                    **results["metadata"],
                }
                rows.append(row)
    return pd.DataFrame(rows)


def get_nth_newest_file(n: int, directory: str):
    files_path = os.path.join(directory, "results/*")
    files = sorted(glob.iglob(files_path), key=os.path.getmtime, reverse=True)
    return files[n]


def print_results_multiple(results: dict[str, dict[str, dict]]):
    for name, result in results.items():
        print_results_single(result, name)


def print_results_single(results: dict[str, dict], title: str):
    # todo: parametrize logging level
    print(HEADER_PRINTOUT.format(title=title))

    print(SUBHEADER_PRINTOUT.format(title="METADATA"))
    for k, v in results["metadata"].items():
        print(f"{k}: {v}")
    print(SUBHEADER_PRINTOUT.format(title="RESULTS"))
    for num, question in results["questions"].items():
        print(f"{question}")
        key = "responses" if "responses" in results else "outputs"
        for i, response in enumerate(results[key][num]):
            print(f"* {i}. {response}")


HEADER_PRINTOUT = "=" * 50 + "\n" + "*" * 5 + "  {title}  " + "*" * 5 + "\n" + "=" * 50
SUBHEADER_PRINTOUT = "-" * 20 + "{title}" + "-" * 20
