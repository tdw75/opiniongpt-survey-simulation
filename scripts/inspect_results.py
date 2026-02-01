import json
import os
import sys

import fire

sys.path.append(os.getcwd())

from src.analysis.results import get_nth_newest_file, print_results_multiple


def main(run_id: str | int = "0", directory: str = "data_files"):
    """Utility script to quickly inspect the results of a simulation run by its ID or by its recency."""
    if str(run_id).isdigit():
        file_name = get_nth_newest_file(int(run_id), directory)
    else:
        file_name = os.path.join(directory, "results", run_id)
    with open(file_name) as f:
        results = json.load(f)
        print_results_multiple(results)


if __name__ == "__main__":
    fire.Fire(main)
