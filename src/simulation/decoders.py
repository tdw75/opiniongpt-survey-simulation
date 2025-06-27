import outlines
import torch
from outlines.models import Transformers
from tqdm import tqdm

from src.prompting.messages import batch_messages, Messages, Prompt, ResponseList
from src.simulation.models import ModelConfig
from src.simulation.utils import get_batches


class BaseDecoder:

    def __init__(self, model, tokenizer, config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def generate(self) -> list[str]:
        raise NotImplementedError

    def simulate_question(
        self,
        question: tuple[Prompt, ResponseList],
        question_flipped: tuple[Prompt, ResponseList],
    ) -> list[str]:
        raise NotImplementedError


class HFDecoder(BaseDecoder):

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
            get_batches(messages_batched, self.config.batch_size), desc="batch"
        ):
            batch_kwargs = self.init_generation_params(batch)
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

    def init_generation_params(self, messages: Messages | list[Messages]):

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


class OutlinesDecoder(BaseDecoder):

    def __init__(self, model, tokenizer, config: ModelConfig):
        super().__init__(model, tokenizer, config)
        self.model = Transformers(self.model, self.tokenizer)

    def simulate_question(
        self,
        question: tuple[Prompt, ResponseList],
        question_flipped: tuple[Prompt, ResponseList],
    ) -> list[str]:
        responses_per_prompt = []

        for prompt, choices in [question, question_flipped]:
            prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            prompt_responses = self.generate(prompt, choices)
            responses_per_prompt.extend(prompt_responses)
        return responses_per_prompt

    def generate(self, prompt: Prompt, choices: ResponseList) -> list[str]:
        generate = outlines.generate.choice(self.model, choices)
        n_remaining = self.config.sample_size // 2
        prompt_responses = []
        while n_remaining > 0:
            n = min(self.config.batch_size, n_remaining)
            batch_responses = generate(prompt, n=n, **self.config.hyperparams)
            prompt_responses.extend(batch_responses)
            n_remaining -= self.config.batch_size
        return prompt_responses
