from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizer


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
) -> dict:
    print(model)
    if by == "respondents":
        responses = simulate_group_of_respondents(
            model, tokenizer, survey, system_prompt, 1000
        )
    elif by == "questions":  # todo: change hardcoded n
        single_question = list(survey.items())[0]
        single_question = {single_question[0]: single_question[1]}
        # todo: change to whole survey, loop through all questions maybe?
        responses = simulate_set_of_responses_multiple_questions(
            model, tokenizer, single_question, system_prompt, hyperparams, 10
        )
    else:
        raise ValueError  # todo: add error message

    return responses


def simulate_single_respondent(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    survey: dict[str, str],
    system_prompt: str,
    hyperparams: dict,
) -> dict[str, str]:

    # todo: function for both OpinionGPT and persona prompting
    # if model == LLaMa: add persona to system prompt or is_persona flag
    text_responses = {}

    previous_responses = """"""

    for number, question in survey.items():
        # todo: add previous_responses to 'assistant' prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": number},
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
    # todo: parametrize active adapter (or outside of function)
    # todo: inject system prompt based on prompting style (e.g. persona, own-history, etc.)

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print("=" * 10, "INPUT", "=" * 10)
    # inputs = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    # todo: change to len of input ids not input object, check if padded - only want length of valid tokens
    input_len = len(inputs)

    # todo: inject hyperparameters/config
    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,  # potentially change as longer answers or not needed/valid (maybe only [1, 30] tokens needed)
        min_new_tokens=4,
        no_repeat_ngram_size=3,
        do_sample=True,
        temperature=1,
    )
    if hyperparams:
        generation_kwargs.update(hyperparams)

    output = model.generate(**generation_kwargs)
    # output = output[input_len:]  # todo: deactivated for debugging
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # todo: remove input with split instead of using length? which is more expensive
    print("-" * 10, "RESPONSE", "-" * 10)
    print(response)
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
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
            model, tokenizer, survey, system_prompt
        )

    return respondents


def simulate_set_of_responses_single_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    hyperparams: dict[str, Any],
    n: int = 1000,
) -> list[str]:

    responses = []
    for i in range(n):
        responses.append(
            simulate_response_single_question(model, tokenizer, messages, hyperparams)
        )

    return responses
