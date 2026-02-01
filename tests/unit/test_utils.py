from src.utils import mark_is_scale_flipped


def test_mark_is_scale_flipped():
    responses = [f"response{i}" for i in range(10)]
    is_flipped = mark_is_scale_flipped(responses)
    assert is_flipped == [False, True] * 5
