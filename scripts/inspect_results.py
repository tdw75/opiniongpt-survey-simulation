import json
import os
import sys

import fire

sys.path.append(os.getcwd())


from src.simulation.utils import get_nth_newest_file, print_results


def main(run_id: str | int = "0", directory: str = "data_files"):

    if run_id.isdigit() or isinstance(run_id, int):
        file_name = get_nth_newest_file(int(run_id), directory)
    else:
        file_name = os.path.join(directory, run_id)
    with open(file_name) as f:
        results = json.load(f)
        print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
