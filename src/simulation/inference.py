import logging

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.prompting.messages import format_messages
from src.simulation.models import ModelConfig

logger = logging.getLogger(__name__)


def simulate_whole_survey(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: dict[str, str],
    system_prompt: str,
) -> dict:
    logger.debug(model)
    if config.aggregation_by == "respondents":
        responses = simulate_group_of_respondents(
            model, tokenizer, config, survey, system_prompt
        )
    elif config.aggregation_by == "questions":
        responses = simulate_set_of_responses_multiple_questions(
            model, tokenizer, config, survey, system_prompt
        )
    else:
        raise ValueError  # todo: add error message

    return responses


def simulate_single_respondent(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: dict[str, str],
    system_prompt: str,
) -> dict[str, str]:

    text_responses = {}

    for number, question in tqdm(survey.items()):  # todo: add desc
        # todo: add previous_responses to 'assistant' prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        generation_kwargs = init_generation_params(tokenizer, config, messages)
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
        config.hyperparams["num_return_sequences"] = config.count

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    )

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
    system_prompt: str,
):
    responses: dict[str, list[str]] = {}

    for number, question in tqdm(
        survey.items(), desc=f"{config.subgroup or 'general'} survey"
    ):
        messages = format_messages(system_prompt, question, config)
        generation_kwargs = init_generation_params(tokenizer, config, messages)
        input_len = generation_kwargs["input_ids"].shape[-1]
        responses[number] = generate_responses(
            model, tokenizer, generation_kwargs, input_len
        )

    return responses


def simulate_group_of_respondents(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: dict[str, str],
    system_prompt: str,
) -> dict[int, dict]:

    respondents = {}
    for i in tqdm(range(config.count), desc="Respondents"):
        respondents[i] = simulate_single_respondent(
            model, tokenizer, config, survey, system_prompt
        )

    return respondents
