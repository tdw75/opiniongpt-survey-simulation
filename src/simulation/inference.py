import logging

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.prompting.messages import batch_messages, Messages
from src.simulation.models import ModelConfig
from src.simulation.utils import get_batches

logger = logging.getLogger(__name__)


def simulate_whole_survey(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: dict[str, str],
    flipped: dict[str, str],
) -> dict[str, list[str]]:
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
        responses = generate_responses(model, tokenizer, generation_kwargs)

        # todo: update messages with assistant response
        text_responses[number] = responses[0]

    return text_responses


def init_generation_params(
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    messages: Messages | list[Messages],
):
    # todo: inject system prompt based on prompting style (e.g. persona, own-history, etc.)

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
    return {**inputs, **config.hyperparams}


def generate_responses(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, generation_kwargs: dict
) -> list[str]:
    """
    function that actually calls the LLM
    """
    input_len = generation_kwargs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
        return tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)


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
    messages_batched = batch_messages([question, question_flipped], config)

    for batch in tqdm(get_batches(messages_batched, config.batch_size), desc="batch"):
        batch_kwargs = init_generation_params(tokenizer, config, batch)
        batch_kwargs["num_return_sequences"] = 1  # todo: might be redundant
        response_batch = generate_responses(model, tokenizer, batch_kwargs)
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
