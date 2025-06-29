import pytest

from models import ModelConfig
from src.simulation.decoders import UnconstrainedDecoder


class TestUnconstrainedDecoder:
    decoder = UnconstrainedDecoder(
        "dummy_model",
        "dummy_tokenizer",
        config=ModelConfig(batch_size=2, decoding_style="unconstrained"),
    )

    @pytest.mark.parametrize("sample_size", (4, 5))
    def test_get_batches(self, sample_size):
        expected_batch1 = [[{"message1": "content"}], [{"message2": "content"}]]
        expected_batch2 = [[{"message3": "content"}], [{"message4": "content"}]]
        expected = [expected_batch1, expected_batch2]

        if sample_size == 5:
            expected.append([[{"message5": "content"}]])

        messages_batched = [
            [{f"message{i}": "content"}] for i in range(1, sample_size + 1)
        ]

        for i, batch in enumerate(self.decoder._get_batches(messages_batched)):
            assert batch == expected[i]
