from enum import Enum

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, PreTrainedTokenizer, \
    PreTrainedModel

bias_to_subreddit = {
    "liberal": "AskALiberal",
    "conservative": "AskConservatives",
    "german": "AskAGerman",
    "american": "AskAnAmerican",
    "latin_american": "AskLatinAmerica",
    "middle_east": "AskMiddleEast",
    "men": "AskMen",
    "women": "AskWomen",
    "people_over_30": "AskPeopleOver30",
    "old_people": "AskOldPeople",
    # "teenager": "AskTeenagers",
}
adapters = list(bias_to_subreddit.keys())


def load_opinion_gpt(device: str = "cuda:2") -> tuple[PreTrainedModel, PreTrainedTokenizer]:

    model_id = "unsloth/Phi-3-mini-4k-instruct"
    lora_id = "HU-Berlin-ML-Internal/opiniongpt-phi3-{adapter}"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    default_adapter = adapters[0]

    model = PeftModel.from_pretrained(
        model, lora_id.format(adapter=default_adapter), adapter_name=default_adapter
    ).to(device)

    for adapter in adapters[1:]:
        # todo: do I need to load all adapters each time?
        print(f"Loading adapter: {adapter}")
        model.load_adapter(lora_id.format(adapter=adapter), adapter)

    # todo: why is the target adapter always German???
    target_adapter = "german"

    if model.active_adapter != "american":
        model.set_adapter(target_adapter)

    return model, tokenizer


def load_llama(device: str = "cuda:2") -> tuple[LlamaForCausalLM, LlamaTokenizer]:

    # todo: add hugging face model id

    model_id = ""
    model = LlamaForCausalLM.from_pretrained(model_id)
    tokenizer = LlamaTokenizer.from_pretrained(model_id)

    return model, tokenizer
