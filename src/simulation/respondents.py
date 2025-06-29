import torch
from outlines.models import Transformers
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from messages import Survey, Messages
from models import ModelConfig


def simulate_single_respondent(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: Survey,
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


def simulate_group_of_respondents(
    model: PreTrainedModel | Transformers,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: Survey,
) -> dict[int, dict]:

    respondents = {}
    for i in tqdm(range(config.sample_size), desc="Respondents"):
        respondents[i] = simulate_single_respondent(model, tokenizer, config, survey)

    return respondents
