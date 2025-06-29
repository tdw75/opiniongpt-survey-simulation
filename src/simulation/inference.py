import logging

from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.prompting.messages import Survey
from src.simulation.decoders import ConstrainedDecoder, UnconstrainedDecoder, BaseDecoder
from src.simulation.models import ModelConfig

logger = logging.getLogger(__name__)

# todo:
#  - load model as transformers(model, tokenizer)
#  - pass survey responses through for choices argument
#  - construct messages in outlines format
#  - init kwargs in outlines format
#  - initialise outlines.generate.choice(llm, choices) before each question
#  - generate with responses = generate(prompts, n=10, sampling=True)


def simulate_whole_survey(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: ModelConfig,
    survey: Survey,
    flipped: Survey,
) -> dict[str, list[str]]:
    logger.debug(model)
    decoder = get_decoder(model, tokenizer, config)
    responses: dict[str, list[str]] = {}
    for number, question in tqdm(survey.items(), desc=decoder.config.run_name):
        responses[number] = decoder.simulate_question(survey[number], flipped[number])
    return responses


def get_decoder(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: ModelConfig
) -> BaseDecoder:
    if config.decoding_style == "constrained":
        return ConstrainedDecoder(model, tokenizer, config)
    elif config.decoding_style == "unconstrained":
        return UnconstrainedDecoder(model, tokenizer, config)
    else:
        raise ValueError(f"Unknown decoding style: {config.decoding_style}")
