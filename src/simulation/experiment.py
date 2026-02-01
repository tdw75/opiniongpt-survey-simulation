import os
from dataclasses import dataclass
from datetime import datetime

import yaml


def generate_run_id(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%Hh%M.%S")
    return f"{timestamp}-{model_name}"


def huggingface_login() -> None:
    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv()
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print("Successfully logged in to Hugging Face!")
    else:
        print("Token is not set. Please save a token in the .env file.")


@dataclass
class Experiment:
    setup: dict
    files: dict
    simulation: dict


def load_experiment(experiment_name: str, root_directory: str) -> Experiment:
    """Load experiment config with base.yaml as default.

    experiment_name: Name of experiment (e.g., 'test' loads experiments/test.yaml)

    Returns: merged config dict (base + experiment overrides)
    """
    # fixme: root_directory is not being fed properly to experiment.files["directory"]
    base_path = os.path.join(root_directory, "experiments", "base.yaml")
    experiment_path = os.path.join(
        root_directory, "experiments", f"{experiment_name}.yaml"
    )

    base = yaml.safe_load(open(base_path))
    experiment = yaml.safe_load(open(experiment_path))

    return Experiment(**_update_config(base, experiment))


def _update_config(base: dict, update: dict) -> dict:
    """Recursively merge update dict into base dict."""
    for k, v in update.items():
        if isinstance(v, dict) and k in base:
            base[k] = _update_config(base[k], v)
        else:
            base[k] = v
    return base
