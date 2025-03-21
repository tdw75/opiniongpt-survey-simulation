from transformers import PreTrainedModel, PreTrainedTokenizer


# todo: model config
# - model_name
# - prompting behaviour (e.g. persona, )
# - aggregation level/style (e.g. simulate one question, simulate whole respondent)

def simulate_whole_survey(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, survey: dict[str, str], by: str
) -> dict:
    if by == "respondents":
        responses = simulate_group_of_respondents(model, tokenizer, 1000, survey)
    elif by == "questions":  # todo: change hardcoded num=5
        single_prompt = list(survey.values())[0]  # todo: change to whole survey
        responses = simulate_set_of_responses_single_question(model, tokenizer, single_prompt, 5)
    else:
        raise ValueError  # todo: add error message

    return responses


def simulate_single_respondent(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, survey: dict[str, str]
) -> dict[str, str]:

    # todo: function for both OpinionGPT and persona prompting
    # if model == LLaMa: add persona prompt or is_persona flag
    text_responses = {}

    previous_responses = """"""

    for question, prompt in survey.items():
        # todo: prompt with previous responses as one run, prepend 'previous_responses' to prompt
        response = simulate_response_single_question(model, tokenizer, prompt)

        # todo: append question number and item stem to 'previous_responses'
        # todo: append response output to 'previous_responses'
        text_responses[question] = response
        # # todo: extract numeric keys for responses

    return text_responses


def simulate_response_single_question(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str
) -> str:
    # todo: parametrize active adapter (or outside of function)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    input_len = len(inputs)  # todo: change to len of input ids not input object, check if padded - only want length of valid tokens

    # todo: inject hyperparameters/config
    generation_kwargs = dict(
        input_ids=inputs,
        max_new_tokens=50,  # potentially change as longer answers or not needed/valid (maybe only [1, 30] tokens needed)
        min_new_tokens=4,
        no_repeat_ngram_size=3,
        do_sample=True,
        temperature=1
    )

    output = model.generate(**generation_kwargs)
    # output = output[input_len:]  # todo: deactivated for debugging
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(response)
    # todo: extract numeric keys for responses (e.g. -1: don't know)
    return response


def simulate_group_of_respondents(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, n_respondents: int, survey: dict[str, str]
) -> dict[int, dict]:

    # todo: allow toggling between persona prompting and opinionGPT
    #  - separate script?? maybe different load_model but same run_inference

    respondents = {}
    for i in range(n_respondents):
        respondents[i] = simulate_single_respondent(model, tokenizer, survey)

    return respondents


def simulate_set_of_responses_single_question(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, n: int = 1000
) -> list[str]:

    responses = []
    for i in range(n):
        responses[i] = simulate_response_single_question(model, tokenizer, prompt)

    return responses
