import re

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
)

bias_to_subreddit = {
    "liberal": "AskALiberal",
    "conservative": "AskConservatives",
    "german": "AskAGerman",
    "american": "AskAnAmerican",
    "latin_america": "AskLatinAmerica",
    "middle_east": "AskMiddleEast",
    "men": "AskMen",
    "women": "AskWomen",
    "people_over_30": "AskPeopleOver30",
    "old_people": "AskOldPeople",
    # "teenager": "AskTeenagers",
}
adapters = list(bias_to_subreddit.keys())


def load_model(
    base_model_name, subgroup: str, is_lora: bool, device: str = "cuda:2"
) -> tuple[PeftModel | PreTrainedModel, PreTrainedTokenizer]:
    model, tokenizer = load_base(base_model_name)
    if is_lora:
        model = load_opinion_gpt(model, device)
        return change_adapter(model, subgroup), tokenizer
    else:
        model = model.to(device)
        # todo: implement persona prompting
        print("No LoRA adapters used")
        return change_persona(model, subgroup), tokenizer


def load_opinion_gpt(model: PreTrainedModel, device: str = "cuda:2") -> PeftModel:

    lora_id = "HU-Berlin-ML-Internal/opiniongpt-phi3-{adapter}"
    default_adapter = adapters[0]

    model = PeftModel.from_pretrained(
        model, lora_id.format(adapter=default_adapter), adapter_name=default_adapter
    ).to(device)

    for adapter in adapters[1:]:  # all adapters loaded and then accessed as needed
        print(f"Loading adapter: {adapter}")
        model.load_adapter(lora_id.format(adapter=adapter), adapter)

    return model


def change_adapter(model: PeftModel, target_adapter: str) -> PeftModel:
    if model.active_adapter != target_adapter:
        model.set_adapter(target_adapter)
        print(f"Changed active adapter to {target_adapter}")
    return model


def load_base(model_id: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_id = MODEL_DIRECTORY.get(model_id, model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if is_phi_model(model_id):
        tokenizer.chat_template = PHI_TOKENIZER_FORMAT
    print(f"Successfully loaded model: {model_id}")
    return model, tokenizer


def change_persona(model, target_persona: str):
    # todo: implement changing personas for LLaMa
    return model


def is_phi_model(model_id: str) -> bool:
    return bool(re.match(r"^.+/phi.*", model_id, re.IGNORECASE))


def default_hyperparams(tokenizer: PreTrainedTokenizer) -> dict:
    return dict(
        max_new_tokens=50,  # potentially change as longer answers or not needed/valid (maybe only [1, 30] tokens needed)
        min_new_tokens=4,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.2,
    )


MODEL_DIRECTORY = {
    "phi": "unsloth/Phi-3-mini-4k-instruct",
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
}


PHI_TOKENIZER_FORMAT = """
{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}
"""
