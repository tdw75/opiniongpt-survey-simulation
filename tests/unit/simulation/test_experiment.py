import pytest

from src.simulation.experiment import load_experiment


@pytest.mark.parametrize(
    "experiment_name, expected",
    [
        (
            "unit_test",
            {
                "experiment_setup": {
                    "random-seed": 43,
                    "directory": "test_data_files",
                    "simulation_name": "unit_test",
                    "variables_file": "sample_variables.csv",
                    "subset_file": "unit_test_subset.json",
                },
                "simulation_params": {
                    "sample_size": 11,
                    "batch_size": 11,
                    "decoding_style": "unconstrained",
                    "base_model": "claude",
                    "temperature": 0.9,
                },
            },
        ),
        (
            "blank",
            {
                "experiment_setup": {
                    "simulation_name": "blank",
                    "random-seed": 43,
                    "directory": "test_data_files",
                    "variables_file": "sample_variables.csv",
                    "subset_file": "final_subset.json",
                },
                "simulation_params": {
                    "sample_size": 2000,
                    "batch_size": 32,
                    "decoding_style": "unconstrained",
                    "base_model": "claude",
                    "temperature": 0.9,
                },
            },
        ),
    ],
)
def test_load_experiment(experiment_name, expected):
    experiment = load_experiment(experiment_name, "test_data_files")
    assert experiment == expected
