import logging

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.prompting.messages import format_messages
from src.simulation.models import ModelConfig
from src.simulation.utils import get_batch

logger = logging.getLogger(__name__)


def simulate_whole_survey(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: dict[str, str],
    flipped: dict[str, str],
) -> dict:
    logger.debug(model)
    if config.aggregation_by == "respondents":
        responses = simulate_group_of_respondents(model, tokenizer, config, survey)
    elif config.aggregation_by == "questions":
        responses = simulate_set_of_responses_multiple_questions(
            model, tokenizer, config, survey, flipped
        )
    else:
        raise ValueError  # todo: add error message

    return responses


def simulate_single_respondent(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: dict[str, str],
) -> dict[str, str]:

    text_responses = {}

    for number, question in tqdm(survey.items()):  # todo: add desc
        # todo: add previous_responses to 'assistant' prompt
        # todo: use format_messages
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": question},
        ]
        generation_kwargs = init_generation_params(tokenizer, config, messages)
        generation_kwargs["num_return_sequences"] = 1
        input_length = generation_kwargs["input_ids"].shape[-1]
        responses = generate_responses(
            model, tokenizer, generation_kwargs, input_length
        )

        # todo: update messages with assistant response
        text_responses[number] = responses[0]

    return text_responses


def init_generation_params(
    tokenizer: PreTrainedTokenizer, config: ModelConfig, messages: list[dict[str, str]]
):
    # todo: inject system prompt based on prompting style (e.g. persona, own-history, etc.)

    if config.aggregation_by == "questions":
        config.hyperparams["num_return_sequences"] = config.batch_size
    elif config.aggregation_by == "respondents":
        config.hyperparams["num_return_sequences"] = 1

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    )
    inputs = {k: v.to(config.device) for k, v in inputs.items()}

    generation_kwargs = dict(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    )
    generation_kwargs.update(config.hyperparams)
    return generation_kwargs


def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    generation_kwargs: dict,
    input_len: int,
) -> list[str]:
    """
    function that actually calls the LLM
    """
    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
        return [
            tokenizer.decode(output[input_len:], skip_special_tokens=True)
            for output in outputs
        ]


def simulate_set_of_responses_multiple_questions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: dict[str, str],
    flipped: dict[str, str],
):
    responses: dict[str, list[str]] = {}
    for number, question in tqdm(survey.items(), desc=config.run_name):
        responses[number] = simulate_set_of_responses_single_question(
            model, tokenizer, config, survey[number], flipped[number]
        )

    return responses


def simulate_set_of_responses_single_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    question: str,
    question_flipped: str,
) -> list[str]:
    responses = []
    idx_to_name = {0: "normal", 1: "flipped"}
    messages = {
        "normal": format_messages(question, config),
        "flipped": format_messages(question_flipped, config)
    }
    generation_kwargs = {
        k: init_generation_params(tokenizer, config, m) for k, m in messages.items()
    }
    input_len = {k: v["input_ids"].shape[-1] for k, v in generation_kwargs.items()}

    for idx, batch in tqdm(enumerate(get_batch(config)), desc="batch"):
        params_batch = generation_kwargs[idx_to_name[idx % 2]]
        params_batch["num_return_sequences"] = batch
        response_batch = generate_responses(
            model, tokenizer, params_batch, input_len[idx_to_name[idx % 2]]
        )
        responses.extend(response_batch)

    return responses


def simulate_group_of_respondents(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: dict[str, str],
) -> dict[int, dict]:

    respondents = {}
    for i in tqdm(range(config.sample_size), desc="Respondents"):
        respondents[i] = simulate_single_respondent(model, tokenizer, config, survey)

    return respondents
