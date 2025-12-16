import re
from typing import Any, Generator, Literal

import outlines
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel

from src.prompting.messages import (
    batch_messages,
    Messages,
    Prompt,
    ResponseList,
    format_messages,
)
from src.data.variables import QNum
from src.simulation.models import ModelConfig


class BaseDecoder:
    """
    Abstract base class for LLM decoders.

    Provides a common interface for simulating responses to survey questions,
    either with unconstrained (free-form) or constrained (format-restricted) decoding.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        """
        Initialise the decoder

        :param model: The underlying language model
        :param tokenizer: The tokeniser corresponding to the model
        :param config: Configuration object with generation parameters
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def generate_responses(self) -> list[str]:
        """
        Call the LLM to generate responses.

        returns: List of generated responses.
        """

        raise NotImplementedError

    def simulate_question(
        self,
        qnum: QNum,
        question: tuple[Prompt, ResponseList],
        question_flipped: tuple[Prompt, ResponseList],
    ) -> list[str]:
        """
        Simulate a set of responses for a single question, using both original and flipped response orderings.

        :param qnum: string containing the question number, e.g. Q1.
        :param question: (prompt, choices) for the original order.
        :param question_flipped: (prompt, choices) for the flipped order.
        :returns: Interleaved list of generated responses.
        """
        raise NotImplementedError


class UnconstrainedDecoder(BaseDecoder):
    """
    Decoder for unconstrained (free-form) generation using HuggingFace models.
    """

    def simulate_question(
        self,
        qnum: QNum,
        question: tuple[Prompt, ResponseList],
        question_flipped: tuple[Prompt, ResponseList],
    ) -> list[str]:
        responses = []
        messages_batched = batch_messages(
            [question[0], question_flipped[0]], self.config
        )

        for batch in tqdm(
            self._get_batches(messages_batched), desc=f"{qnum}-batch", leave=False
        ):
            batch_kwargs = self._init_generation_params(batch)
            batch_kwargs["num_return_sequences"] = 1  # todo: might be redundant
            response_batch = self.generate_responses(batch_kwargs)
            responses.extend(response_batch)

        return responses

    def generate_responses(self, generation_kwargs: dict) -> list[str]:
        """
        Generate a batch of responses from the HuggingFace model.

        :param generation_kwargs: Keyword arguments for model.generate().
        :returns: List of generated responses.
        """
        input_len = generation_kwargs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)
            return self.tokenizer.batch_decode(
                outputs[:, input_len:], skip_special_tokens=True
            )

    def _init_generation_params(self, messages: Messages | list[Messages]):
        """
        Prepare generation parameters for HuggingFace model.

        :param messages: Messages to format as prompt.
        :returns: Generation parameters including input tensors and hyperparameters.
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        return {**inputs, **self.config.hyperparams}

    def _get_batches(
        self, messages: list[Messages]
    ) -> Generator[list[Messages], Any, None]:
        """
        Yield batches of messages for generation.

        :param messages: List of messages to batch.
        :yields: Next batch of messages.
        """
        for i in range(0, len(messages), self.config.batch_size):
            yield messages[i : i + self.config.batch_size]


class ConstrainedDecoder(BaseDecoder):
    """
    Transformer decoder that restricts the format of responses using constrained decoding with 'outlines'.
    """

    def __init__(
        self,
        model: PreTrainedModel,  # huggingface object
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        """
        Initialize the constrained decoder with Outlines.

        :param model: The underlying language HuggingFace model.
        :param tokenizer: The tokeniser corresponding to the model.
        :param config: Configuration object with generation parameters.
        """
        super().__init__(model, tokenizer, config)
        self.llm = outlines.from_transformers(self.model, self.tokenizer)

    def simulate_question(
        self,
        qnum: QNum,
        question: tuple[Prompt, ResponseList],
        question_flipped: tuple[Prompt, ResponseList],
    ) -> list[str]:
        """
        Simulate responses for both original and flipped prompt orderings using constrained decoding.

        :param question: Tuple (prompt, choices) for the original order.
        :param question_flipped: Tuple (prompt, choices) for the flipped order.
        :returns: Interleaved list of generated responses.
        """
        prompts = [self._prepare_inputs(pr) for pr, _ in [question, question_flipped]]
        # choices_list = [
        #     self._prepare_choices(qnum, ch) for _, ch in [question, question_flipped]
        # ]
        choices_list = [question[1], question_flipped[1]]
        responses_per_prompt = [
            self.generate_responses(
                prompt, choices, f"{qnum}-batch-{'orig'if i==0 else 'flipped'}"
            )
            for i, (prompt, choices) in enumerate(zip(prompts, choices_list))
        ]
        return self._interleave(responses_per_prompt)

    def generate_responses(
        self, prompt: Prompt, choices: ResponseList, desc: str
    ) -> list[str]:
        """
        Generate a batch of responses from the model using Outlines constrained decoding.

        :param prompt: The formatted prompt string.
        :param choices: regex for valid response choices for constrained decoding.
        :returns: List of generated responses.
        """
        generator = outlines.Generator(self.llm, Literal[*choices])
        prompt_responses = []
        for n in tqdm(self._get_batch_sizes(), desc=desc, leave=False):
            batch_responses = generator([prompt] * n, **self.config.hyperparams)
            prompt_responses.extend(batch_responses)

        return prompt_responses

    def _prepare_inputs(self, prompt: Prompt) -> str:
        """
        Format the prompt using the tokeniser's chat template for constrained decoding.

        :param prompt: The user prompt to format.
        :returns: The formatted prompt string.
        """
        messages = format_messages(prompt, self.config)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _prepare_choices(qnum: QNum, choices: ResponseList) -> str:
        """
        Build a regex pattern to match choices with optional prefixes and whitespace.

        :param qnum: The question number (used as a prefix).
        :param choices: List of valid choice strings (e.g., ["1: agree", "2: not sure"]).
        :returns: Regex pattern string for use with outlines.
        """
        prefixes = [r"your response:", r"response:", rf"{qnum}:"]
        prefix_pattern = r"(?:" + "|".join([re.escape(p) for p in prefixes]) + r")?\s*"
        patterns = [rf"\s*{prefix_pattern}{re.escape(choice)}\s*" for choice in choices]
        return r"(?i)" + "|".join(patterns)

    def _get_batch_sizes(self) -> list[int]:
        """
        Compute the batch sizes for sampling, ensuring memory efficiency.
        note: if config.sample_size is odd then actual number of outputs will be config.sample_size - 1

        :returns: List of integer batch sizes to use for generation.
        """
        total = self.config.sample_size // 2
        batch_size = min(self.config.batch_size, total)
        last_batch_size = total % batch_size
        last_batch = [last_batch_size] if last_batch_size > 0 else []
        return [batch_size] * (total // batch_size) + last_batch

    @staticmethod
    def _interleave(responses_per_prompt: list[list[str]]) -> list[str]:
        """
        Interleave responses from multiple prompts to match the desired output order.

        :param responses_per_prompt: List of lists, each containing responses for a prompt.
        :returns: Interleaved flat list of responses.
        """
        return [resp for pair in zip(*responses_per_prompt) for resp in pair]
