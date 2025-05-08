import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.prompting.messages import format_messages
from src.simulation.models import ModelConfig


def simulate_whole_survey(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: dict[str, str],
    system_prompt: str,
    num: int = 10,  # todo: remove after debugging
) -> dict:
    print(model)
    if config.aggregation_by == "respondents":
        responses = simulate_group_of_respondents(
            model, tokenizer, config, survey, system_prompt, num
        )
    elif config.aggregation_by == "questions":  # todo: change hardcoded n
        responses = simulate_set_of_responses_multiple_questions(
            model, tokenizer, config, survey, system_prompt, num
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

    for number, question in survey.items():
        # todo: add previous_responses to 'assistant' prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        response = simulate_response_single_question(model, tokenizer, config, messages)
        # todo: update messages with assistant response
        text_responses[number] = response

    return text_responses


def simulate_response_single_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    messages: list[dict[str, str]],
) -> str:
    """
    function that actually calls the LLM
    """
    # todo: parametrize active adapter (or outside of function)
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
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[-1]

    generation_kwargs = dict(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    )
    generation_kwargs.update(config.hyperparams)

    with torch.no_grad():
        output = model.generate(**generation_kwargs)
        response = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
        # todo: extract numeric keys for responses (e.g. -1: don't know)
    return response


def simulate_set_of_responses_multiple_questions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: dict[str, str],
    system_prompt: str,
    n: int = 1000,
):

    responses: dict[str, list[str]] = {}

    for number, question in survey.items():  # todo: add tqdm
        # todo: hardcoded as phi, change to use LLaMa
        messages = format_messages(system_prompt, question, config)
        responses[number] = simulate_set_of_responses_single_question(
            model, tokenizer, config, messages, n
        )

    return responses


def simulate_group_of_respondents(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: dict[str, str],
    system_prompt: str,
    n_respondents: int = 1000,
) -> dict[int, dict]:

    # todo: allow toggling between persona prompting and opinionGPT
    #  - separate script?? maybe different load_model but same run_inference

    respondents = {}
    for i in range(n_respondents):  # todo: add tqdm
        respondents[i] = simulate_single_respondent(
            model, tokenizer, config, survey, system_prompt
        )

    return respondents


def simulate_set_of_responses_single_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    messages: list[dict[str, str]],
    n: int = 1000,
) -> list[str]:

    responses = []
    for i in range(n):

        response = simulate_response_single_question(model, tokenizer, config, messages)
        responses.append(response)

    return responses
