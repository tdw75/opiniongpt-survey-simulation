from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.simulation.models import default_hyperparams


# todo: model config
# - model_name
# - prompting behaviour (e.g. persona, )
# - aggregation level/style (e.g. simulate one question, simulate whole respondent)


def simulate_whole_survey(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    survey: dict[str, str],
    by: str,
    system_prompt: str,
    hyperparams: dict[str, Any],
    num: int = 10,  # todo: remove after debugging
) -> dict:
    print(model)
    if by == "respondents":
        responses = simulate_group_of_respondents(
            model, tokenizer, survey, system_prompt, num
        )
    elif by == "questions":  # todo: change hardcoded n
        responses = simulate_set_of_responses_multiple_questions(
            model, tokenizer, survey, system_prompt, hyperparams, num
        )
    else:
        raise ValueError  # todo: add error message

    return responses


def simulate_single_respondent(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    survey: dict[str, str],
    system_prompt: str,
    hyperparams: dict = None,
) -> dict[str, str]:

    # todo: function for both OpinionGPT and persona prompting
    # if model == LLaMa: add persona to system prompt or is_persona flag
    text_responses = {}

    previous_responses = """"""

    for number, question in survey.items():
        # todo: add previous_responses to 'assistant' prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        response = simulate_response_single_question(
            model, tokenizer, messages, hyperparams
        )
        # todo: update previous_responses with question, number and response
        text_responses[number] = response
        # todo: extract numeric keys for responses

    return text_responses


def simulate_response_single_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    hyperparams: dict[str, Any] = None,
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
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[-1]

    # todo: inject hyperparameters/config
    generation_kwargs = dict(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    )

    hyperparams = {**default_hyperparams(tokenizer), **(hyperparams or {})}
    generation_kwargs.update(hyperparams)

    with torch.no_grad():
        output = model.generate(**generation_kwargs)
        response = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
        # todo: extract numeric keys for responses (e.g. -1: don't know)
    return response


def simulate_set_of_responses_multiple_questions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    survey: dict[str, str],
    system_prompt: str,
    hyperparams: dict[str, Any],
    n: int = 1000,
):

    responses: dict[str, list[str]] = {}

    for number, question in survey.items():  # todo: add tqdm
        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{system_prompt}\n{question}"},
        ]
        responses[number] = simulate_set_of_responses_single_question(
            model, tokenizer, messages, hyperparams, n
        )

    return responses


def simulate_group_of_respondents(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    survey: dict[str, str],
    system_prompt: str,
    n_respondents: int = 1000,
) -> dict[int, dict]:

    # todo: allow toggling between persona prompting and opinionGPT
    #  - separate script?? maybe different load_model but same run_inference

    respondents = {}
    for i in range(n_respondents):  # todo: add tqdm
        respondents[i] = simulate_single_respondent(
            model, tokenizer, survey, system_prompt,
        )

    return respondents


def simulate_set_of_responses_single_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    hyperparams: dict[str, Any],
    n: int = 1000,
) -> list[str]:

    print("=" * 10, "INPUT", "=" * 10)
    for m in messages:
        print(m["role"], " prompt")
        print(m["content"])

    responses = []
    for i in range(n):

        response = simulate_response_single_question(
            model, tokenizer, messages, hyperparams
        )
        responses.append(response)
        print("-" * 10, f"RESPONSE {i}", "-" * 10)
        print(response)

    return responses
