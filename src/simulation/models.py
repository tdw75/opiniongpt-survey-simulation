from enum import Enum

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, PreTrainedTokenizer, \
    PreTrainedModel

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


def load_opinion_gpt(device: str = "cuda:2", model_id: str = "unsloth/Phi-3-mini-4k-instruct") -> tuple[PreTrainedModel, PreTrainedTokenizer]:

    lora_id = "HU-Berlin-ML-Internal/opiniongpt-phi3-{adapter}"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    default_adapter = adapters[0]

    model = PeftModel.from_pretrained(
        model, lora_id.format(adapter=default_adapter), adapter_name=default_adapter
    ).to(device)

    for adapter in adapters[1:]:  # all adapters loaded and then accessed as needed
        print(f"Loading adapter: {adapter}")
        model.load_adapter(lora_id.format(adapter=adapter), adapter)

    return model, tokenizer


def change_adapter(model: PeftModel, target_adapter: str) -> PeftModel:
    if model.active_adapter != target_adapter:
        model.set_adapter(target_adapter)
    return model


def load_llama(device: str = "cuda:2", model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> tuple[LlamaForCausalLM, LlamaTokenizer]:

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer


def change_persona(model, target_persona: str):
    # todo: implement changing personas for LLaMa
    raise NotImplementedError
