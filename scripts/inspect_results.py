import json
import os
import sys

import fire

sys.path.append(os.getcwd())


from src.simulation.utils import get_nth_newest_file, print_results


def main(run_id: str | int = 0, directory: str = "data_files"):

    if isinstance(run_id, str):
        file_name = run_id
    elif isinstance(run_id, int):
        file_name = get_nth_newest_file(run_id, directory)
    else:
        raise TypeError(f"run_id is type {type(run_id)} but must be either int or str")
    with open(os.path.join(directory, file_name)) as f:
        results = json.load(f)
        print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
