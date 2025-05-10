import fire

from src.simulation.models import adapters, ModelConfig
from src.simulation.run import run_single
from src.simulation.utils import (
    huggingface_login,
    print_results,
    generate_run_id,
    save_results,
    get_run_name,
)


def main(
    base_model_name: str = "phi",
    directory: str = "data_files",
    filename: str = "variables.csv",
    question_format: str = "individual",
    device: str = "cuda:2",
    question_num: int = 0,
    **kwargs,  # LLM hyperparams
):
    instruct_config = ModelConfig(
        base_model_name=base_model_name,
        subgroup=None,
        is_lora=False,
        is_persona=False,
        device=device,
        aggregation_by="questions",
        hyperparams=kwargs,
    )
    run_id = generate_run_id(base_model_name)
    run_name = get_run_name(base_model_name, True, None)
    simulated_surveys = {  # todo: add personas
        run_name: run_single(
            instruct_config,
            directory,
            run_id,
            filename,
            question_format,
            question_num,
            **kwargs,
        )
    }
    print("=" * 50, "\n", "-" * 20, run_name, "-" * 20, "\n", "=" * 50)
    print_results(simulated_surveys[run_name])

    opinion_gpt_config = ModelConfig(
        base_model_name=base_model_name,
        is_lora=True,
        is_persona=False,
        device=device,
        aggregation_by="questions",
        hyperparams=kwargs,
    )
    for subgroup in adapters:
        run_name = get_run_name(base_model_name, True, subgroup)
        opinion_gpt_config.change_subgroup(subgroup)
        simulated_surveys[run_name] = run_single(
            opinion_gpt_config,
            directory,
            run_id,
            filename,
            question_format,
            question_num,
            **kwargs,
        )
        print("=" * 50, "\n", "-" * 20, run_name, "-" * 20, "\n", "=" * 50)
        print_results(simulated_surveys[run_name])

    save_results(simulated_surveys, directory, run_id)


if __name__ == "__main__":
    huggingface_login()
    fire.Fire(main)
