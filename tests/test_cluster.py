import json
import os

import fire
import numpy as np
import pandas as pd


class MockModel:

    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        return x * self.temperature


def load_data(directory: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(directory, "mock_model_data.csv"), index_col=0)


def save_data(directory: str, data: dict) -> None:
    with open(os.path.join(directory, "survey_run.json"), "w") as f:
        json.dump(data, f)


def save_metadata(directory: str, metadata: dict) -> None:
    with open(os.path.join(directory, "survey_metadata.json"), "w") as f:
        json.dump(metadata, f)


def test_run(directory: str = None, temperature: float = 1):

    directory = directory or "test_data_files/mock_data"

    model = MockModel(temperature)
    data = load_data(directory)
    preds = pd.DataFrame(model(data.values), columns=data.columns)
    save_data(directory, {"responses": preds.to_dict(orient="list")})
    save_metadata(directory, {"temp": model.temperature})


def generate_test_data(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
    df = pd.DataFrame({"q1": range(10), "q2": range(10, 0, -1)})
    df.to_csv(os.path.join(directory, "mock_model_data.csv"))


if __name__ == "__main__":
    fire.Fire(test_run)
