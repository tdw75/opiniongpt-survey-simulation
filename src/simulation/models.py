import logging
import re
from typing import Literal, Any

from peft import PeftModel
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from src.prompting.system import (
    build_survey_context_message,
    build_survey_context_for_persona,
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


MODEL_DIRECTORY = {
    "phi": "unsloth/Phi-3-mini-4k-instruct",
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
}

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    base_model_name: Literal[tuple(MODEL_DIRECTORY.keys())] = "phi"
    subgroup: Literal[tuple(adapters + [None])] = None
    is_lora: bool = False
    is_persona: bool = False
    device: str = "cuda:2"
    aggregation_by: Literal["questions", "respondent"] = "questions"
    decoding_style: Literal["constrained", "unconstrained"] = "unconstrained"
    sample_size: int = 500
    batch_size: int = 50
    hyperparams: dict = {}
    system_prompt: str = None

    def model_post_init(self, context: Any, /) -> None:
        default_hyperparams: dict[str, Any] = dict(
            max_new_tokens=16, top_p=0.9, temperature=0.6, do_sample=True
        )
        self.hyperparams = {**default_hyperparams, **self.hyperparams}
        self.system_prompt = self.system_prompt or build_survey_context_message()

    @property
    def model_type(self) -> str:
        return "opinion-gpt" if self.is_lora else f"instruct"

    @property
    def model_id(self) -> str:
        return MODEL_DIRECTORY.get(self.base_model_name, self.base_model_name)

    @property
    def is_phi_model(self):
        return bool(re.match(r"^.+/phi.*", self.model_id, re.IGNORECASE))

    def change_subgroup(self, subgroup: str):
        if subgroup in adapters + [None]:
            self.subgroup = subgroup
        else:
            raise ValueError(f"Invalid subgroup: {subgroup}")

    @property
    def run_name(self):
        persona_suffix = "with-persona" if self.is_persona else "no-persona"
        return f"{self.base_model_name}-{self.model_type}-{self.subgroup or 'general'}-{persona_suffix}"


def load_model(
    config: ModelConfig,
) -> tuple[PeftModel | PreTrainedModel, PreTrainedTokenizer]:
    model, tokenizer = load_base(config)
    if config.is_lora:
        return load_opinion_gpt(model, config), tokenizer
    else:
        logger.info("No LoRA adapters used")
        return model.to(config.device), tokenizer


def load_opinion_gpt(model: PreTrainedModel, config: ModelConfig) -> PeftModel:

    lora_id = "HU-Berlin-ML-Internal/opiniongpt-phi3-{adapter}"
    default_adapter = adapters[0]

    model = PeftModel.from_pretrained(
        model, lora_id.format(adapter=default_adapter), adapter_name=default_adapter
    ).to(config.device)

    for adapter in adapters[1:]:  # all adapters loaded to be accessed as needed
        logger.info(f"Loading adapter: {adapter}")
        model.load_adapter(lora_id.format(adapter=adapter), adapter)

    return model


def load_base(config: ModelConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, padding_side="left")
    # if is_phi_model(model_id):
    #     tokenizer.chat_template = PHI_TOKENIZER_FORMAT
    logger.info(f"Successfully loaded model: {config.model_id}")
    return model, tokenizer


def change_subgroup(
    model: PreTrainedModel | PeftModel, config: ModelConfig, new_subgroup: str
) -> tuple[PreTrainedModel | PeftModel, ModelConfig]:
    config.change_subgroup(new_subgroup)
    if config.is_lora:
        model = change_adapter(model, config.subgroup)
    if config.is_persona:
        config.system_prompt = build_survey_context_for_persona(config.subgroup)
    return model, config


def change_adapter(model: PeftModel, target_adapter: str) -> PeftModel:
    if model.active_adapter != target_adapter:
        model.set_adapter(target_adapter)
        logger.info(f"Changed active adapter to {target_adapter}")
    return model


PHI_TOKENIZER_FORMAT = """
{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}
"""
