from typing import Any, Generator

import torch
from outlines import Generator as OutlinesGenerator
from outlines.models import Transformers
from tqdm import tqdm

from src.prompting.messages import batch_messages, Messages, Prompt, ResponseList
from src.simulation.models import ModelConfig


class BaseDecoder:

    def __init__(self, model, tokenizer, config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def generate_responses(self) -> list[str]:
        raise NotImplementedError

    def simulate_question(
        self,
        question: tuple[Prompt, ResponseList],
        question_flipped: tuple[Prompt, ResponseList],
    ) -> list[str]:
        raise NotImplementedError


class UnconstrainedDecoder(BaseDecoder):

    def simulate_question(
        self,
        question: tuple[Prompt, ResponseList],
        question_flipped: tuple[Prompt, ResponseList],
    ) -> list[str]:
        responses = []
        messages_batched = batch_messages(
            [question[0], question_flipped[0]], self.config
        )

        for batch in tqdm(
            self._get_batches(messages_batched), desc="batch", leave=False
        ):
            batch_kwargs = self._init_generation_params(batch)
            batch_kwargs["num_return_sequences"] = 1  # todo: might be redundant
            response_batch = self.generate_responses(batch_kwargs)
            responses.extend(response_batch)

        return responses

    def generate_responses(self, generation_kwargs: dict) -> list[str]:
        """
        function that actually calls the LLM
        """
        input_len = generation_kwargs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)
            return self.tokenizer.batch_decode(
                outputs[:, input_len:], skip_special_tokens=True
            )

    def _init_generation_params(self, messages: Messages | list[Messages]):

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
        for i in range(0, len(messages), self.config.batch_size):
            yield messages[i : i + self.config.batch_size]


class ConstrainedDecoder(BaseDecoder):

    def __init__(self, model, tokenizer, config: ModelConfig):
        super().__init__(model, tokenizer, config)
        self.model = Transformers(self.model, self.tokenizer)

    def simulate_question(
        self,
        question: tuple[Prompt, ResponseList],
        question_flipped: tuple[Prompt, ResponseList],
    ) -> list[str]:

        prompts = [
            self._prepare_inputs(question[0]),
            self._prepare_inputs(question_flipped[0]),
        ]
        choices_list = [question[1], question_flipped[1]]
        responses_per_prompt = [
            self.generate_responses(prompt, choices)
            for prompt, choices in zip(prompts, choices_list)
        ]
        return self._interleave(responses_per_prompt)

    def generate_responses(self, prompt: Prompt, choices: ResponseList) -> list[str]:
        generator = OutlinesGenerator(self.model, choices)
        prompt_responses = []
        for n in tqdm(self._get_batch_sizes(), desc="batch", leave=False):
            batch_responses = generator(prompt, n=n, **self.config.hyperparams)
            prompt_responses.extend(batch_responses)

        return prompt_responses

    def _prepare_inputs(self, prompt: Prompt) -> str:
        return self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

    def _get_batch_sizes(self) -> list[int]:
        total = self.config.sample_size // 2
        batch_size = self.config.batch_size
        last_batch_size = total % batch_size
        last_batch = [last_batch_size] if last_batch_size > 0 else []
        return [batch_size] * (total // batch_size) + last_batch

    @staticmethod
    def _interleave(responses_per_prompt: list[list[str]]) -> list[str]:
        return [resp for pair in zip(*responses_per_prompt) for resp in pair]
