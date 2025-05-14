import glob
import json
import os

import pandas as pd


def load_survey_results(file_name: str, directory: str) -> dict[str, dict]:
    path = os.path.join(directory, "results", file_name)
    with open(path) as f:
        return json.load(f)


def survey_results_to_df(survey_results: dict[str, dict], variables: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subgroup, results in survey_results.items():
        for num, responses in results["responses"].items():
            variable = variables[variables["number"]==num]
            for response in responses:
                row = {
                    "number": num,
                    "group": variable["group"].item(),
                    "subtopic": variable["subtopic"].item(),
                    "question": results["questions"][num],
                    "response": response,
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
        for i, response in enumerate(results["responses"][num]):
            print(f"* {i}. {response}")


HEADER_PRINTOUT = "=" * 50 + "\n" + "*" * 5 + "  {title}  " + "*" * 5 + "\n" + "=" * 50
SUBHEADER_PRINTOUT = "-" * 20 + "{title}" + "-" * 20
